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

from dataset.database import ImageDatabase, TextDatabase
from dataset.triplet_data import TripletIm2Text
from models.models import TripletNetIm2Text # , EmbeddingNetImage, EmbeddingNetText
from models.resnet import EmbeddingNetImage, EmbeddingNetText
from utils import losses
from utils import metrics
from utils import trainer
from utils import test
from utils.early_stopper import EarlyStopper
from sklearn.metrics import precision_recall_curve

import argparse
import json
import ujson
import joblib
import os
import functools 
import wandb

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_V2_Weights


from utils import losses
from utils import trainer
from utils.early_stopper import EarlyStopper





def train(args):   
    if args.sweep:
        wandb.init(project="m5-w4", entity="grup7")
        # IMPORTANT PUT HERE THE NAME OF VARIABLES THAT YOU WANT TO SWEEP
        args.margin = wandb.config.margin
        args.dim_out_fc = wandb.config.dim_out_fc
        args.learning_rate = wandb.config.lr
    else:
        wandb=None
    
    
    
    print('Margin: ', args.margin)
    print('Dim out fc: ', args.dim_out_fc)

    name = 'task_a' + 'RESNET_dim_out_fc_' + args.dim_out_fc + '_margin_' + str(args.margin) + '_lr_' + str(args.learning_rate)
    
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

    output_path = os.path.join(env_path, 'results/task_a', name)


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


    triplet_train_dataset = TripletIm2Text(train_annot_path, train_path, transform=transform)
    # triplet_test_dataset = TripletIm2Text(val_annot_path, val_path, transform=transform)
    
    transform = torch.nn.Sequential(
                FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(),
                transforms.Resize((256, 256)),
            )


    image_val_dataset = ImageDatabase(val_annot_path, val_path, num_samples=100, transform=transform)
    text_val_dataset = TextDatabase(val_annot_path, val_path, num_samples= 100, transform=transform)
    

    # ------------------------------- DATALOADER --------------------------------


    triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=True, num_workers=10)
    triplet_test_loader = None

    image_val_loader = DataLoader(image_val_dataset, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=True, num_workers=10)
    text_val_loader = DataLoader(text_val_dataset, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=True, num_workers=10)
    
    

    # ------------------------------- MODEL --------------------------------
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    margin = args.margin
    weights_image = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    weights_text = args.weights_text
        
    # Pretrained model from torchvision or from checkpoint
    embedding_net_image = EmbeddingNetImage(weights=weights_image,  dim_out_fc = args.dim_out_fc).to(device)
    embedding_net_text = EmbeddingNetText(weights=weights_text, device=device,  dim_out_fc = args.dim_out_fc).to(device)

    model = TripletNetIm2Text(embedding_net_image, embedding_net_text).to(device)

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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1, last_epoch=-1)
    # lr_scheduler = None

    log_interval = 10

    trainer.fit(args, triplet_train_loader, triplet_test_loader, image_val_dataset, image_val_loader, text_val_dataset, text_val_loader, model, loss_func, optimizer, lr_scheduler, num_epochs,
                device, log_interval, output_path, name='img2txt', wandb = wandb)


    # --------------------------------- TESTING --------------------------------
    image_test_dataset = ImageDatabase(val_annot_path, val_path, num_samples=2000, transform=transform)
    text_test_dataset = TextDatabase(val_annot_path, val_path, num_samples= 2000, transform=transform)
    
    image_test_loader = DataLoader(image_val_dataset, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=True, num_workers=10)
    text_test_loader = DataLoader(text_val_dataset, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=True, num_workers=10)
    
    output_path = os.path.join(output_path, 'test')
    # Create output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    test.test_resnet(args, model, image_test_dataset, image_test_loader, text_test_dataset, text_test_loader, output_path, device=device, wandb=wandb, retrieval=True)
    



if __name__ == '__main__':
    
    # ------------------------------- ARGS ---------------------------------
    parser = argparse.ArgumentParser(description='Task A')
    parser.add_argument('--resnet_type', type=str, default='V1', help='Resnet version (V1 or V2)')
    parser.add_argument('--dim_out_fc', type=str, default='as_image', help='Dimension of the output of the fully connected layer (as_image or as_text)')
    parser.add_argument('--train', type=bool, default=True, help='Train or test')
    parser.add_argument('--sweep', type=bool, default=True, help='Sweep or not')
    parser.add_argument('--weights_model', type=str,
                        default='/ghome/group03/M5-Project/week4/checkpoints/best_loss_task_a_finetunning.h5',
                        help='Path to weights')
    parser.add_argument('--weights_text', type=str,
                        default='/ghome/group03/M5-Project/week5/utils/text/fasttext_wiki.en.bin',
                        help='Path to weights of text model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--gpu', type=int, default=5, help='GPU device id')
    args = parser.parse_args()
    
    if args.sweep:
        assert args.train == True, "Sweep only works for training"
        sweep_config = {
            'name': 'task_a_RESNET_asText_sweep_2',
            'method': 'grid',
            'parameters':{
                'margin': {
                    'values': [0.001, 1, 10]
                    # 'value': 1
                },
                'dim_out_fc': {
                    # 'values': ['as_image', 'as_text']
                    'value': 'as_text'
                },
                'lr': {
                    'value': 1e-4
                }
            }
        }
        
        
        sweep_id = wandb.sweep(sweep=sweep_config, project="m5-w5", entity="grup7")
        
        wandb.agent(sweep_id, function=functools.partial(train, args))
    
    else:
        train(args)