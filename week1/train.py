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


def train(args):
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")
        scaler = torch.cuda.amp.GradScaler()
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")
    else:
        print("CPU is available")
        device = torch.device("cpu")
    # Initialize wandb
    wandb.init(mode=args.wandb)

    # tf.random.set_seed(42)
    # np.random.seed(42)

    # Print wandb.config
    print(wandb.config)

    args.experiment_name = wandb.config.experiment_name

    # Load the model
    model = ResNet().to(device)
    # model = torch.compile(model) # Pytorch 2.0

    # Write model summary to console and WandB
    wandb.config.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", wandb.config.num_params)
    summary(model, (3, wandb.config.IMG_HEIGHT, wandb.config.IMG_WIDTH))

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

    train_dataset = MITDataset(data_dir='/ghome/group03/mcv/m3/datasets/MIT_small_train_1', split_name='train',
                               transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.BATCH_SIZE, shuffle=True, num_workers=8)

    val_dataset = MITDataset(data_dir='/ghome/group03/mcv/m3/datasets/MIT_small_train_1', split_name='test',
                             transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.BATCH_SIZE, shuffle=False, num_workers=8)

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.LEARNING_RATE,
                                 weight_decay=wandb.config.WEIGHT_DECAY)

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
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #  if (i + 1) % 10 == 0:
            train_acc = accuracy(outputs, labels)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, wandb.config.EPOCHS, i + 1, len(train_loader), loss.item(), train_acc * 100))

            wandb.log({"epoch": epoch, "train_loss": loss.item()})
            wandb.log({"epoch": epoch, "train_accuracy": train_acc})
            wandb.log({"epoch": epoch, "learning_rate": wandb.config.LEARNING_RATE})

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for j, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss += loss_fn(outputs, labels)
                val_acc += accuracy(outputs, labels)

            val_loss = val_loss / (j + 1)
            val_acc = val_acc / (j + 1)
            wandb.log({"epoch": epoch, "val_loss": val_loss})
            wandb.log({"epoch": epoch, "val_accuracy": val_acc})
            print('Epoch [{}/{}], Val_Loss: {:.4f}, Val_Accuracy: {:.2f}%'
                  .format(epoch + 1, wandb.config.EPOCHS, val_loss.item(), val_acc * 100))

        #  # Learning rate scheduler
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
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_val_loss': best_val_loss,
                         'best_val_acc': best_val_acc,
                         'optimizer': optimizer.state_dict(),
                         }, is_best_loss, is_best_acc, filename=wandb.config.experiment_name + '.h5')

        if is_best_loss or is_best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())

        t1 = time.time()
        total_time += t1 - t0
        print("Epoch time: ", t1 - t0)
        print("Total time: ", total_time)

    model.load_state_dict(best_model_wts)
