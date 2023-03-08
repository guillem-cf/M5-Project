import torch
import numpy as np
import os
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau



from utils.metrics import accuracy, top_k_acc
from utils.early_stopper import EarlyStopper
from utils.checkpoint import save_checkpoint
from dataset.mit import MITDataset
from models.resnet import ResNet

from torchsummary import summary
from torchviz import make_dot
import graphviz
import copy


# import matplotlib
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.utils import plot_model
# import tensorflow_addons as tfa
# from wandb.keras import WandbCallback
# from model import MyModel
# from utils import save_plots, get_data_train, get_data_validation, get_data_test, get_optimizer

# import visualkeras

# import tensorflow as tf

# matplotlib.use("Agg")

# print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)



def train(args):
    if torch.cuda.is_available() == False:
        print("CUDA is not available")
        exit()

    # Initialize wandb
    wandb.init(mode=args.wandb)
    
    # tf.random.set_seed(42)
    # np.random.seed(42)

    # Print wandb.config
    print(wandb.config)

    args.experiment_name = wandb.config.experiment_name

    # Load the model
    model = ResNet().cuda()

    # Write model summary to console and WandB
    wandb.config.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", wandb.config.num_params)
    summary(model, (3, wandb.config.IMG_HEIGHT, wandb.config.IMG_WIDTH,))

    # # plot model architecture graph to file and WandB
    # graph = make_dot(model.mean(), params=dict(model.named_parameters()))
    # graph.format = 'png'
    # graph.render(filename='./images/model_' + wandb.config.experiment_name)
    # wandb.log({'Model Architecture Graph': wandb.Image(filename='./images/model_' + wandb.config.experiment_name)})

    # # plot layered view of model architecture to file and WandB
    # dot = make_dot(model.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True, show_dtype=True)
    # dot.format = 'png'
    # dot.render(filename='./network_draw/' + wandb.config.experiment_name)
    # wandb.log({'Layered View of Model Architecture': wandb.Image(filename='./network_draw/' + wandb.config.experiment_name)})


    # Load the data
    transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Resize((wandb.config.IMG_HEIGHT, wandb.config.IMG_WIDTH))])

    # Data augmentation
    if wandb.config.data_augmentation == True:
        transform = transforms.Compose(
                        [transforms.RandomHorizontalFlip(),
                         transforms.RandomAffine(degrees=0, shear=10, translate=(0.1, 0.1)),
                         transforms.ToTensor(),
                         transforms.Resize((wandb.config.IMG_HEIGHT, wandb.config.IMG_WIDTH))])
    
    train_dataset = MITDataset(data_dir = '/ghome/group03/mcv/m3/datasets/MIT_small_train_1', split_name='train', transform=transform) 
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.BATCH_SIZE, shuffle=True, num_workers=8)

    val_dataset = MITDataset(data_dir = '/ghome/group03/mcv/m3/datasets/MIT_small_train_1', split_name='test', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.BATCH_SIZE, shuffle=False, num_workers=8)


    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.LEARNING_RATE)

    # Early stoppper
    early_stopper = EarlyStopper(patience=50, min_delta=10)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_val_acc = 0
    total_time = 0
    for epoch in range(wandb.config.EPOCHS):
        t0 = time.time()

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i + 1) % 10 == 0:
            train_acc = accuracy(outputs, labels)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                .format(epoch + 1, wandb.config.EPOCHS, i + 1, len(train_loader), loss.item(), train_acc*100))
            
            wandb.log({"train_loss": loss.item()})
            wandb.log({"train_accuracy": train_acc})
            wandb.log({"learning_rate": wandb.config.LEARNING_RATE})

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for j, (images, labels) in enumerate(val_loader):
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                val_loss += loss_fn(outputs, labels)
                val_acc += accuracy(outputs, labels)

            val_loss = val_loss/(j+1)
            val_acc = val_acc/(j+1)
            wandb.log({"val_loss": val_loss})
            wandb.log({"val_accuracy": val_acc})
            print('Epoch [{}/{}], Val_Loss: {:.4f}, Val_Accuracy: {:.2f}%'
                    .format(epoch + 1, wandb.config.EPOCHS, val_loss.item(), val_acc*100))

        # # Learning rate scheduler
        lr_scheduler.step(val_loss)

        # Early stopping
        if early_stopper.early_stop(val_loss):
            print("Early stopping at epoch: ", epoch)
            break

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best_loss = True
        else:
            is_best_loss = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            is_best_acc = True
        else:
            is_best_acc = False
        save_checkpoint({ 'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_val_loss': best_val_loss,
                        'best_val_acc': best_val_acc,
                        'optimizer' : optimizer.state_dict(),
                        }, is_best_loss, is_best_acc, filename = wandb.config.experiment_name + '.h5')
        
        if is_best_loss or is_best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())


        t1 = time.time()
        total_time += t1 - t0
        print("Epoch time: ", t1 - t0)
        print("Total time: ", total_time)

    model.load_state_dict(best_model_wts)
    return 


    # defining the early stop criteria
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=20, mode="auto", min_lr=1e-8)
    # saving the best model based on val_loss
    mc1 = ModelCheckpoint('./checkpoint/best_' + wandb.config.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_loss', mode='min', save_best_only=True)
    mc2 = ModelCheckpoint('./checkpoint/best_' + wandb.config.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_accuracy', mode='max', save_best_only=True)


    optimizer = get_optimizer(wandb.config.OPTIMIZER)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    
    if wandb.config.CALLBACKS:
        wandb_callback = WandbCallback(input_type="images", labels=["coast", "forest", "highway", "inside_city", "mountain", "Opencountry", "street", "tallbuilding"],
                                   output_type="label", training_data=get_data_train(), validation_data=get_data_validation(), log_weights=True, log_gradients=True, log_evaluation=True, log_batch_frequency=10)
    else:
        wandb_callback = WandbCallback()
        
    history = model.fit(
        get_data_train(),
        steps_per_epoch=(int(400 // wandb.config.BATCH_SIZE) + 1),
        epochs=wandb.config.EPOCHS,
        validation_data=get_data_validation(),
        validation_steps=(int(wandb.config.VALIDATION_SAMPLES // wandb.config.BATCH_SIZE) + 1),
        callbacks=[wandb_callback, mc1, mc2, es, reduce_lr], 
        workers=24
    )
    result = model.evaluate(get_data_test())
    print(result)
    print(history.history.keys())
    save_plots(history, args)
