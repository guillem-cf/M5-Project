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

from dataset.triplet_data import CocoDatasetWeek5 as TripletCOCODataset
from models.models import TripletNet
from models.models import ObjectEmbeddingNet
from utils import losses
from utils import trainer
from utils.early_stopper import EarlyStopper

# ------------------------------- PATHS --------------------------------
env_path = os.path.dirname(os.path.abspath(__file__))
# get path of current file
dataset_path = '/ghome/group03/mcv/datasets/COCO'
# dataset_path = '../../datasets/COCO'

output_path = os.path.join(env_path, 'Results/task_a')

# Create output path if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# -------------------------------- DEVICE --------------------------------
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

# ------------------------------- DATASET --------------------------------
train_path = os.path.join(dataset_path, 'train2014')
val_path = os.path.join(dataset_path, 'val2014')

# train_annot_path = os.path.join(dataset_path, 'instances_train2014.json')
# val_annot_path = os.path.join(dataset_path, 'instances_val2014.json')

object_image_dict = json.load(open(os.path.join(dataset_path, 'mcv_image_retrieval_annotations.json')))

try:
    print('Loading train negative image dict')
    path = os.path.join(dataset_path, 'train_dict_negative_img_low.json')
    with open(path, 'r') as f:
        train_negative_image_dict = ujson.load(f)
    print('Done!')
    
    # print('Loading val negative image dict')
    # path = os.path.join(dataset_path, 'val_dict_negative_img_low.json')
    # with open(path, 'r') as f:
    #     val_negative_image_dict = ujson.load(f)
    # print('Done!')
except:
    train_negative_image_dict = None
    # val_negative_image_dict = None



transform = torch.nn.Sequential(
            FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(),
            transforms.Resize((256, 256)),
        )

print(object_image_dict.keys())

triplet_train_dataset = TripletCOCODataset(os.path.join(dataset_path, "captions_train2014.json"), train_path, transform=transform)
# triplet_test_dataset = TripletCOCODataset(None, object_image_dict, val_path, split_name='val',
#                                           dict_negative_img=val_negative_image_dict, transform=transform)

# ------------------------------- DATALOADER --------------------------------


triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=128, shuffle=True,
                                  pin_memory=True, num_workers=10)

triplet_test_loader = None

# ------------------------------- MODEL --------------------------------
num_epochs = 1
learning_rate = 1e-5
margin = 0.1
pretrained = False

weights = ResNet50_Weights
    
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
