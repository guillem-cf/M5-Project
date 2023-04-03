import argparse
import os
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
)

from dataset.siamese_data import SiameseMITDataset
from models.models import SiameseNet, EmbeddingNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from utils.checkpoint import save_checkpoint_loss
from utils.early_stopper import EarlyStopper
from utils import metrics, trainer, losses
import torch.nn.functional as F

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


if __name__ == '__main__':
    # ------------------------------- ARGUMENTS --------------------------------
    parser = argparse.ArgumentParser(description='Task B')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--weights', type=str, default='/ghome/group03/M5-Project/week4/checkpoints/best_loss_task_a_finetunning.h5', help='Path to weights')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    args = parser.parse_args()

    # ------------------------------- PATHS --------------------------------
    env_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(env_path, 'mcv/datasets/MIT_split')
   
    output_path = os.path.join(env_path, 'M5-Project/week4/Results/Task_b')
    
    # Create output path if it does not exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    
    # -------------------------------- DEVICE --------------------------------
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
        
    
    # ------------------------------- DATASET --------------------------------
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transforms.Compose([
                                 ResNet50_Weights.IMAGENET1K_V2.transforms(),
                                ]))
    
    test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transforms.Compose([
                                 ResNet50_Weights.IMAGENET1K_V2.transforms(),
                                ]))
    
    
    
    siamese_train_dataset = SiameseMITDataset(train_dataset, split_name='train')
    siamese_test_dataset = SiameseMITDataset(test_dataset, split_name='test')
    
    
    # ------------------------------- DATALOADER --------------------------------
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, **kwargs)
    
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    siamese_train_loader = DataLoader(siamese_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    siamese_test_loader = DataLoader(siamese_test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    # ------------------------------- MODEL --------------------------------
    margin = 1.
    
    # Pretrained model from torchvision or from checkpoint
    if args.pretrained:
        embedding_net = EmbeddingNet(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
        
    # else:
    #     weights_model = torch.load(args.weights)['model_state_dict']
    #     embedding_net = EmbeddingNet(weights=weights_model).to(device)
    
    model = SiameseNet(embedding_net).to(device)
    
    #--------------------------------- TRAINING --------------------------------  
    
    # Loss function
    loss_func = losses.ContrastiveLoss().to(device)  # margin
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Early stoppper
    early_stopper = EarlyStopper(patience=50, min_delta=10)

    # Learning rate scheduler
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)
    
    log_interval = 5
    
    trainer.fit(siamese_train_loader, siamese_test_loader, model, loss_func, optimizer, lr_scheduler, args.num_epochs, device, log_interval, output_path)

    # Plot emmbeddings
    train_embeddings_cl, train_labels_cl = metrics.extract_embeddings(train_loader, model, device)
    path = os.path.join(output_path, 'train_embeddings.png')
    metrics.plot_embeddings(train_embeddings_cl, train_labels_cl, path)

    val_embeddings_cl, val_labels_cl = metrics.extract_embeddings(test_loader, model, device)
    path = os.path.join(output_path, 'val_embeddings.png')
    metrics.plot_embeddings(val_embeddings_cl, val_labels_cl, path)

    metrics.tsne_features(train_embeddings_cl, train_labels_cl, "train", labels=test_dataset.classes, output_dir="Results/Task_b")
    metrics.tsne_features(val_embeddings_cl, val_labels_cl, "test", labels=test_dataset.classes, output_dir="Results/Task_b")
    