import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from dataset.triplet_data import TripletMITDataset
from models.models import TripletNet, EmbeddingNet
from utils import losses
from utils import metrics
from utils import trainer
from utils.early_stopper import EarlyStopper
from sklearn.metrics import precision_recall_curve

import argparse
import json
import ujson
import joblib
import os
import functools 

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_V2_Weights

from dataset.triplet_data import TripletIm2Text
from models.models import TripletNet
from models.models import ObjectEmbeddingNet
from utils import losses
from utils import trainer
from utils.early_stopper import EarlyStopper

# ------------------------------- ARGS ---------------------------------
parser = argparse.ArgumentParser(description='Task E')
parser.add_argument('--resnet_type', type=str, default='V1', help='Resnet version (V1 or V2)')
parser.add_argument('--weighted', type=bool, default=True, help='Weighted features')
parser.add_argument('--fc', type=bool, default=False, help='Use fully connected layer')
parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
parser.add_argument('--weights', type=str,
                    default='/ghome/group03/M5-Project/week4/checkpoints/best_loss_task_a_finetunning.h5',
                    help='Path to weights')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
parser.add_argument('--gpu', type=int, default=7, help='GPU device id')
args = parser.parse_args()

# -------------------------------- GPU --------------------------------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
    torch.cuda.amp.GradScaler()
elif torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device("cpu")
else:
    print("CPU is available")
    device = torch.device("cpu")
    
# ------------------------------- PATHS --------------------------------
env_path = os.path.dirname(os.path.abspath(__file__))
# get path of current file
dataset_path = '/ghome/group03/mcv/datasets/COCO'
# dataset_path = '../../datasets/COCO'

output_path = os.path.join(env_path, 'Results/task_a')

# Create output path if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)


# ------------------------------- DATASET --------------------------------
train_path = os.path.join(dataset_path, 'train2014')
val_path = os.path.join(dataset_path, 'val2014')

train_annot_path = os.path.join(dataset_path, 'captions_train2014.json')
val_annot_path = os.path.join(dataset_path, 'captions_val2014.json')


transform = torch.nn.Sequential(
            FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(),
            transforms.Resize((256, 256)),
        )


triplet_train_dataset = TripletIm2Text(os.path.join(dataset_path, train_annot_path), train_path, transform=transform)
# triplet_test_dataset = TripletCOCODataset(None, object_image_dict, val_path, split_name='val',
#                                           dict_negative_img=val_negative_image_dict, transform=transform)

# ------------------------------- DATALOADER --------------------------------


triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=128, shuffle=True,
                                  pin_memory=True, num_workers=10)

triplet_test_loader = None

# ------------------------------- MODEL --------------------------------
num_epochs = args.num_epochs
learning_rate = args.learning_rate
margin = args.margin
weights = FasterRCNN_ResNet50_FPN_Weights
    
# Pretrained model from torchvision or from checkpoint
embedding_net = EmbeddingNet(weights=weights).to(device)

model = TripletNet(embedding_net).to(device)

# Set all parameters to be trainable
for param in model.parameters():
    param.requires_grad = True
    
# Print the number of trainable parameters
print('Number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# --------------------------------- TRAINING --------------------------------

# Loss function
loss_func = losses.TripletLoss(margin).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Early stoppper
early_stopper = EarlyStopper(patience=50, min_delta=10)

# Learning rate scheduler
# lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)
lr_scheduler = None

log_interval = 1

trainer.fit(triplet_train_loader, triplet_test_loader, model, loss_func, optimizer, lr_scheduler, num_epochs,
            device, log_interval, output_path, name='task_a')
