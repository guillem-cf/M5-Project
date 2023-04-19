import argparse
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from dataset.triplet_data import TripletIm2Text
from models.models import TripletNetIm2Text, EmbeddingNetImage, EmbeddingNetText
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
    wandb.init(project="m5-w4", entity="grup7")
    
    args.margin = wandb.config.margin
    args.dim_out_fc = wandb.config.dim_out_fc
    
    print('Margin: ', args.margin)
    print('Dim out fc: ', args.dim_out_fc)

    name = 'task_a' + '_dim_out_fc_' + args.dim_out_fc + '_margin_' + str(args.margin)
    
    # -------------------------------- GPU --------------------------------
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

    output_path = os.path.join(env_path, 'Results/task_a', name)


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
    triplet_test_dataset = TripletIm2Text(val_annot_path, val_path, transform=transform)

    # ------------------------------- DATALOADER --------------------------------


    #triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                pin_memory=True, num_workers=10)

    # triplet_test_loader = DataLoader(triplet_test_dataset, batch_size=args.batch_size, shuffle=True,
    #                                 pin_memory=True, num_workers=10)
    triplet_test_loader = None

    # ------------------------------- MODEL --------------------------------

    weights_image = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    weights_text = args.weights_text
        
    # Pretrained model from torchvision or from checkpoint
    embedding_net_image = EmbeddingNetImage(weights=weights_image).to(device)
    embedding_net_text = EmbeddingNetText(weights=weights_text, device=device).to(device)

    model = TripletNetIm2Text(embedding_net_image, embedding_net_text).to(device)

   
# --------------------------------- INFERENCE --------------------------------
    
    print("Calculating val database embeddings...")
    start = time.time()
    val_embeddings = metrics.extract_embeddings_coco(db_val_loader, model, device)
    end = time.time()
    print("Time to calculate val database embeddings: ", end - start)
    if val_embeddings.shape[1] == 2:
        path = os.path.join(output_path, 'val_embeddings.png')
        metrics.plot_embeddings_coco(val_embeddings, None, None, 'Validation Embeddings', path)

# --------------------------------- RETRIEVAL ---------------------------------
    
 
    knn = knn.fit(all_txt_features, all_txt_labels)
    neighbors = knn.kneighbors(all_img_features, return_distance=False)
    predictions = all_txt_labels[neighbors]

    p1 = mpk(all_img_labels, predictions, 1)
    p5 = mpk(all_img_labels, predictions, 5)
    

    
    
    # Compute positive and negative values
    evaluation = metrics.positives_coco(neighbors, db_dataset_train, db_dataset_val)

    
# --------------------------------- METRICS ---------------------------------
    metrics.calculate_APs_coco(evaluation, output_path)
    
    metrics.plot_PR_binary(evaluation, output_path)
    






if __name__ == '__main__':
    
    # ------------------------------- ARGS ---------------------------------
    parser = argparse.ArgumentParser(description='Task A')
    parser.add_argument('--resnet_type', type=str, default='V1', help='Resnet version (V1 or V2)')
    parser.add_argument('--dim_out_fc', type=str, default='as_image', help='Dimension of the output of the fully connected layer (as_image or as_text)')
    parser.add_argument('--train', type=bool, default=True, help='Train or test')
    parser.add_argument('--weights_model', type=str,
                        default='/ghome/group03/M5-Project/week5/Results/task_a/task_a_dim_out_fc_as_text_margin_1/task_a_triplet_10.pth',
                        help='Path to weights')
    parser.add_argument('--weights_text', type=str,
                        default='/ghome/group03/M5-Project/week5/Results/task_a/task_a_dim_out_fc_as_text_margin_1/task_a_triplet_10.pth',
                        help='Path to weights of text model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--gpu', type=int, default=7, help='GPU device id')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    args = parser.parse_args()
    
    sweep_config = {
        'name': 'task_a_margin_sweep',
        'method': 'grid',
        'parameters':{
            'margin': {
                'values': [1, 10, 50, 100]
                # 'value': 1
            },
            'dim_out_fc': {
                'values': ['as_image', 'as_text']
                # 'value': 'as_image'
            }
        }
    }
    
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="m5-w5", entity="grup7")
    
    wandb.agent(sweep_id, function=functools.partial(train, args))