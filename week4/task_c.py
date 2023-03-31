import argparse
import os
import time

import torch
import torchvision.transforms as transforms
from dataset.triplet_data import TripletMITDataset
from models.tripletResnet import TripletResNet
from pytorch_metric_learning import losses
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from utils.early_stopper import EarlyStopper

from week1.utils.checkpoint import save_checkpoint_loss

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
        model = TripletResNet(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    else:
        model = TripletResNet().to(device)
    loss_func = losses.TripletMarginLoss(margin=args.margin).to(device)

    # Load the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    train_dataset = TripletMITDataset(data_dir=dataset_path, split_name='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = TripletMITDataset(data_dir=dataset_path, split_name='test', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Early stoppper
    early_stopper = EarlyStopper(patience=50, min_delta=10)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    train_loss_list = []
    val_loss_list = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    total_time = 0
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        model.train()
        loop = tqdm(train_loader)
        for idx, img_triplet in enumerate(loop):
            anchor_img, pos_img, neg_img, anchor_target, pos_target, neg_target = img_triplet
            anchor_img, pos_img, neg_img, anchor_target, pos_target, neg_target = (
                anchor_img.to(device),
                pos_img.to(device),
                neg_img.to(device),
                anchor_target.to(device),
                pos_target.to(device),
                neg_target.to(device),
            )

            # ponemos a cero los gradientes
            optimizer.zero_grad()
            # Backprop (calculamos todos los gradientes automáticamente)
            loss.backward()
            # update de los pesos
            optimizer.step()
            loop.set_description(f"Train: Epoch [{epoch}/{args.num_epochs}]")
            loop.set_postfix(loss=loss.item())

        print({"epoch": epoch, "train_loss": loss.item()})

        train_loss_list.append(loss.item())

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            loop = tqdm(val_loader)
            for idx, img_triplet in enumerate(loop):
                anchor_img, pos_img, neg_img, anchor_target, pos_target, neg_target = img_triplet
                anchor_img, pos_img, neg_img, anchor_target, pos_target, neg_target = (
                    anchor_img.to(device),
                    pos_img.to(device),
                    neg_img.to(device),
                    anchor_target.to(device),
                    pos_target.to(device),
                    neg_target.to(device),
                )

                val_loss += loss_func(embeddings, labels)
                loop.set_description(f"Validation: Epoch [{epoch}/{args.num_epochs}]")
                loop.set_postfix(val_loss=val_loss.item())

            val_loss = val_loss / (idx + 1)
            print({"epoch": epoch, "val_loss": val_loss})

            val_loss_list.append(float(val_loss))

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