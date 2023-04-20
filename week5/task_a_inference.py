import argparse
import os
import time
import numpy as np
from sklearn.impute import KNNImputer
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
from utils.retrieval import extract_retrieval_examples_img2text
from utils.early_stopper import EarlyStopper
from dataset.database import ImageDatabase, TextDatabase
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier

import argparse
import json
import ujson
import joblib
import os
import functools 
import wandb
import umap
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_V2_Weights


from utils import losses
from utils import trainer
from utils.early_stopper import EarlyStopper






if __name__ == '__main__':

    # ------------------------------- ARGS ---------------------------------
    parser = argparse.ArgumentParser(description='Task A')
    parser.add_argument('--resnet_type', type=str, default='V1', help='Resnet version (V1 or V2)')
    parser.add_argument('--dim_out_fc', type=str, default='as_image', help='Dimension of the output of the fully connected layer (as_image or as_text)')
    parser.add_argument('--train', type=bool, default=True, help='Train or test')
    parser.add_argument('--weights_model', type=str,
                        default='/ghome/group03/M5-Project/week5/Results/task_a_old/task_a_dim_out_fc_as_image_margin_1_lr_0.0001/task_a_triplet_10.pth',
                        help='Path to weights')
    parser.add_argument('--weights_text', type=str,
                        default='/ghome/group03/M5-Project/week5/utils/text/fasttext_wiki.en.bin',
                        help='Path to weights of text model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--gpu', type=int, default=7, help='GPU device id')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    args = parser.parse_args()
    

    # -------------------------------- GPU --------------------------------
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    """
    os.environ['OPENBLAS_NUM_THREADS'] = '32'
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1
    """

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

    output_path = os.path.dirname(args.weights_model)


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


    image_dataset = ImageDatabase(val_annot_path, val_path, transform=transform)
    text_dataset = TextDatabase(val_annot_path, val_path, transform=transform)

    # ------------------------------- DATALOADER --------------------------------


    image_loader = DataLoader(image_dataset, batch_size=128, shuffle=False,
                                    pin_memory=True, num_workers=10)

    text_loader = DataLoader(text_dataset, batch_size=128, shuffle=False,
                                pin_memory=True, num_workers=10)


    # ------------------------------- MODEL --------------------------------

    weights_image = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    weights_text = args.weights_text
        
    # Pretrained model from torchvision or from checkpoint
    embedding_net_image = EmbeddingNetImage(weights=weights_image).to(device)
    embedding_net_text = EmbeddingNetText(weights=weights_text, device=device).to(device)  

    model = TripletNetIm2Text(embedding_net_image, embedding_net_text).to(device)

    model.load_state_dict(torch.load(args.weights_model, map_location=device))



    # --------------------------------- INFERENCE --------------------------------
    
    if args.dim_out_fc == 'as_image':
        dim_features = 3840
    elif args.dim_out_fc == 'as_text':
        dim_features = 1000

    print("Calculating image val database embeddings...")
    start = time.time()
    val_embeddings_image, labels_image = metrics.extract_embeddings_image(image_loader, model, device, dim_features = dim_features) # dim_features = 300)
    end = time.time()
    print("Time to calculate image val database embeddings: ", end - start)

    if val_embeddings_image.shape[1] == 2:
        path = os.path.join(output_path, 'val_embeddings.png')
        metrics.plot_embeddings_coco(val_embeddings_image, None, None, 'Validation Embeddings', path)


    print("Calculating text val database embeddings...")
    start = time.time()
    val_embeddings_text, labels_text = metrics.extract_embeddings_text(text_loader, model, device, dim_features = dim_features) # dim_features = 300)
    end = time.time()
    print("Time to calculate text val database embeddings: ", end - start)

    if val_embeddings_image.shape[1] == 2:
        path = os.path.join(output_path, 'val_embeddings.png')
        metrics.plot_embeddings_coco(val_embeddings_text, None, None, 'Validation Embeddings', path)


    # --------------------------------- RETRIEVAL ---------------------------------
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=32) 
    
    print("Fitting KNN...")
    start = time.time()
    # knn = knn.fit(val_embeddings_text, range(len(val_embeddings_text)))
    knn = knn.fit(val_embeddings_text, labels_text)
    end = time.time()

    print("Calculating KNN...")
    start = time.time()
    neighbors = knn.kneighbors(val_embeddings_image, return_distance=False)
    end = time.time()
    print("Time to calculate KNN: ", end - start)
    
    # Map the indices of the neighbors matrix to their corresponding 'id' values
    id_neighbors_matrix = np.vectorize(lambda i: labels_text[i])(neighbors)
    
    # Compute positive and negative values
    evaluation = metrics.positives_ImageToText(neighbors, id_neighbors_matrix, text_dataset, image_dataset)


    # --------------------------------- METRICS ---------------------------------
    metrics.calculate_APs_coco(evaluation, output_path)

    metrics.plot_PR_binary(evaluation, output_path)


    # --------------------------------- PLOT ---------------------------------
    
    extract_retrieval_examples_img2text(neighbors, id_neighbors_matrix, databaseDataset=text_dataset, queryDataset=image_dataset, output_path=output_path)
    
    #umap
    
    # reducer = umap.UMAP(random_state=42)
    # reducer.fit(val_embeddings_image)
    # umap_image_embeddings = reducer.transform(val_embeddings_image)
    # umap_text_embeddings = reducer.transform(val_embeddings_text)

    # metrics.plot_embeddings_ImageText(umap_image_embeddings, umap_text_embeddings, "UMAP Train", "Results/umap_embeddings.png")
