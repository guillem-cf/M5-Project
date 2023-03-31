import argparse
import os
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from dataset.siamese_data import SiameseMITDataset
from models.models import SiameseResNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from utils.checkpoint import save_checkpoint_loss
from utils.early_stopper import EarlyStopper
from utils import losses
import torch.nn.functional as F

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task B')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    args = parser.parse_args()

    # current path
    current_path = os.getcwd()
    dataset_path = os.path.join(current_path, '../../dataset/MIT_split')

    # device
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")
        torch.cuda.amp.GradScaler()
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")
    else:
        print("CPU is available")
        device = torch.device("cpu")

    if args.pretrained:
        model = SiameseResNet(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    else:
        model = SiameseResNet().to(device)
    loss_func = losses.ContrastiveLoss().to(device)  # margin

    # Load the data
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, shear=0, translate=(0, 0.1)),
            transforms.ToTensor(),
            transforms.Resize((64, 64), antialias=False),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SiameseMITDataset(data_dir=dataset_path, split_name='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = SiameseMITDataset(data_dir=dataset_path, split_name='test', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Early stoppper
    early_stopper = EarlyStopper(patience=50, min_delta=10)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []


    # best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    total_time = 0
    for epoch in range(1, args.num_epochs + 1):
        num_correct = 0
        num_total = 0
        t0 = time.time()
        model.train()
        loop = tqdm(train_loader)
        for idx, (img1, img2, label) in enumerate(loop):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            # Forward pass
            E1, E2 = model(img1, img2)
            # labels must be a 1D tensor of shape (batch_size,)
            loss = loss_func(E1, E2, label)
            dist_E1_E2 = F.pairwise_distance(E1, E2, 2)

            pred = dist_E1_E2 < 0.5
            pred = pred.type(torch.FloatTensor).to(device)
            num_correct += torch.sum(pred == label).item()
            num_total += label.size(0)
            train_accuracy = num_correct / num_total

            # ponemos a cero los gradientes
            optimizer.zero_grad()
            # Backprop (calculamos todos los gradientes automáticamente)
            loss.backward()
            # update de los pesos
            optimizer.step()
            loop.set_description(f"Train: Epoch [{epoch}/{args.num_epochs}]")
            loop.set_postfix(loss=loss.item(), accuracy=train_accuracy)

        print({"epoch": epoch, "val_loss": loss.item(), "train_accuracy": train_accuracy})

        train_loss_list.append(loss.item())
        train_acc_list.append(train_accuracy)

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            loop = tqdm(val_loader)
            for idx, img_triplet in enumerate(loop):
                num_correct = 0
                num_total = 0
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                # Forward pass
                E1, E2 = model(img1, img2)
                # labels must be a 1D tensor of shape (batch_size,)
                val_loss += loss_func(E1, E2, label)
                dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
                pred = dist_E1_E2 < 0.5
                pred = pred.type(torch.FloatTensor).to(device)
                num_correct += torch.sum(pred == label).item()
                num_total += label.size(0)
                val_accuracy = num_correct / num_total
                loop.set_description(f"Validation: Epoch [{epoch}/{args.num_epochs}]")
                loop.set_postfix(val_loss=val_loss.item(), val_accuracy=val_accuracy)

            val_loss = val_loss / (idx + 1)
            print({"epoch": epoch, "val_loss": val_loss.item(), "val_accuracy": val_accuracy})

            val_loss_list.append(float(val_loss))
            val_acc_list.append(float(val_accuracy))

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

        if is_best_loss:
            print("Best model saved at epoch: ", epoch, " with val_loss: ", best_val_loss.item())
            save_checkpoint_loss(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict(),
                },
                is_best_loss,
                filename="task_b_siamese" + '.h5',
            )

        t1 = time.time()
        total_time += t1 - t0
        print("Epoch time: ", t1 - t0)
        print("Total time: ", total_time)


output_path = os.path.join(current_path, 'Results/Task_b/')
if not os.path.exists(output_path):
    os.makedirs(output_path)
torch.save(model.state_dict(), output_path + "Task_b_siamese.pth")

plot_step = 100

plt.figure(figsize=(10, 12), dpi=150)
plt.title("Loss during training", size=18)
plt.plot(
    np.arange(0, args.num_epochs, plot_step), train_loss_list[0::plot_step], color="blue", linewidth=2.5, label="Train subset"
)
plt.plot(
    np.arange(0, args.num_epochs, plot_step), val_loss_list[0::plot_step], color="orange", linewidth=2.5, label="Val subset"
)
plt.xticks(np.arange(0, args.num_epochs, plot_step).astype(int))
plt.xlabel("Epoch", size=12)
plt.ylabel("Loss", size=12)
plt.legend()
plt.savefig("Results/Task_b/plot_loss.png")
plt.close()

plt.figure(figsize=(10, 12), dpi=150)
plt.title("Accuracy during training", size=18)
plt.plot(
    np.arange(0, args.num_epochs, plot_step), train_acc_list[0::plot_step], color="blue", linewidth=2.5, label="Train subset"
)
plt.plot(np.arange(0, args.num_epochs, plot_step), val_acc_list[0::plot_step], color="orange", linewidth=2.5, label="Val subset")
plt.xticks(np.arange(0, args.num_epochs, plot_step).astype(int))
plt.xlabel("Epoch", size=12)
plt.ylabel("Accuracy", size=12)
plt.legend()
plt.savefig("Results/Task_b/plot_accuracy.png")
plt.close()
