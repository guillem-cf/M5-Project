import argparse
import json
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from dataset.triplet_data import TripletCOCODataset
from models.models import TripletNet_fasterRCNN, ObjectEmbeddingNet
from utils import losses
from utils import trainer
from utils import metrics
from utils.early_stopper import EarlyStopper


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def triplet_collate_fn(batch):
    anchor_images = []
    positive_images = []
    negative_images = []
    anchor_boxes = []
    anchor_labels = []
    positive_boxes = []
    positive_labels = []
    negative_boxes = []
    negative_labels = []

    for item in batch:
        # Unpack data and target from item
        (anchor_img, positive_img, negative_img), target = item

        # Append images to lists
        anchor_images.append(anchor_img)
        positive_images.append(positive_img)
        negative_images.append(negative_img)

        # Unpack target
        anchor_boxes_, anchor_labels_, positive_boxes_, positive_labels_, negative_boxes_, negative_labels_ = target

        # Append bounding boxes and labels to lists
        anchor_boxes.append(anchor_boxes_)
        anchor_labels.append(anchor_labels_)
        positive_boxes.append(positive_boxes_)
        positive_labels.append(positive_labels_)
        negative_boxes.append(negative_boxes_)
        negative_labels.append(negative_labels_)

    # Stack images into tensors
    anchor_images = torch.stack(anchor_images)
    positive_images = torch.stack(positive_images)
    negative_images = torch.stack(negative_images)

    # Return tuple of tensors
    return (anchor_images, positive_images, negative_images), (
    anchor_boxes, anchor_labels, positive_boxes, positive_labels, negative_boxes, negative_labels)


if __name__ == '__main__':
    # ------------------------------- ARGUMENTS --------------------------------
    parser = argparse.ArgumentParser(description='Task E')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--weights', type=str,
                        default='/ghome/group03/M5-Project/week4/checkpoints/best_loss_task_a_finetunning.h5',
                        help='Path to weights')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    args = parser.parse_args()

    # ------------------------------- PATHS --------------------------------
    env_path = os.path.dirname(os.path.abspath(__file__))
    # get path of current file
    dataset_path = os.path.join(env_path, '../../datasets/COCO')

    output_path = os.path.join(env_path, './Results/Task_e')

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
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_path = os.path.join(dataset_path, 'train2014')
    val_path = os.path.join(dataset_path, 'val2014')

    train_annot_path = os.path.join(dataset_path, 'instances_train2014.json')
    val_annot_path = os.path.join(dataset_path, 'instances_val2014.json')

    object_image_dict = json.load(open(os.path.join(dataset_path, 'mcv_image_retrieval_annotations.json')))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # scale to range [0,1]
    ])

    train_dataset = CocoDetection(root=train_path, annFile=train_annot_path)
    val_dataset = CocoDetection(root=val_path, annFile=val_annot_path)
    triplet_train_dataset = TripletCOCODataset(train_dataset, object_image_dict, train_path, split_name='train',
                                               transform=transform)
    triplet_test_dataset = TripletCOCODataset(val_dataset, object_image_dict, val_path, split_name='val',
                                              transform=transform)

    # ------------------------------- DATALOADER --------------------------------
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, **kwargs)

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=triplet_collate_fn, **kwargs)
    triplet_test_loader = DataLoader(triplet_test_dataset, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=triplet_collate_fn, **kwargs)

    # ------------------------------- MODEL --------------------------------
    margin = 1.

    # Pretrained model from torchvision or from checkpoint
    if args.pretrained:
        embedding_net = ObjectEmbeddingNet(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                                           num_classes=len(train_dataset.coco.cats)+1).to(device)

    model = TripletNet_fasterRCNN(embedding_net).to(device)

    # --------------------------------- TRAINING --------------------------------

    # Loss function
    loss_func = losses.TripletLoss(margin).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Early stoppper
    early_stopper = EarlyStopper(patience=50, min_delta=10)

    # Learning rate scheduler
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)

    log_interval = 5

    trainer.fit(triplet_train_loader, triplet_test_loader, model, loss_func, optimizer, lr_scheduler, args.num_epochs,
                device, log_interval, output_path, name='task_e')

    # Plot emmbedings
    train_embeddings_cl, train_labels_cl = metrics.extract_embeddings(train_loader, model, device)
    path = os.path.join(output_path, 'train_embeddings.png')
    metrics.plot_embeddings(train_embeddings_cl, train_labels_cl, path)
    val_embeddings_cl, val_labels_cl = metrics.extract_embeddings(val_loader, model, device)
    path = os.path.join(output_path, 'val_embeddings.png')
    metrics.plot_embeddings(val_embeddings_cl, val_labels_cl, path)

    metrics.tsne_features(train_embeddings_cl, train_labels_cl, "train", labels=val_dataset.classes, output_dir="Results/Task_e")
    metrics.tsne_features(val_embeddings_cl, val_labels_cl, "test", labels=val_dataset.classes, output_dir="Results/Task_e")
