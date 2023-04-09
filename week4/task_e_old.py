import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.coco import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Import losses for faster rcnn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Define the COCO classes
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def train_one_epoch(model, writer, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    # summary = tensorboard.Epoch# summary(writer, epoch, prefix='train')

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Log the loss and learning rate to tensorboard
        # summary.add_scalar('loss', losses.item())
        # summary.add_scalar('lr', optimizer.param_groups[0]["lr"])

        if i % print_freq == 0:
            print(f"Epoch {epoch}, iteration {i}: loss = {losses.item()}")

    # summary.flush()


def evaluate(model, writer, data_loader, device, epoch, print_freq):
    model.eval()
    # summary = tensorboard.Epoch# summary(writer, epoch, prefix='val')

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)

            # get the evaluation metrics
            loss_dict = output['loss_classifier'] + output['loss_box_reg'] + output['loss_objectness'] + output[
                'loss_rpn_box_reg']
            losses = sum(loss for loss in loss_dict.values())

            # Log the loss to tensorboard
            # summary.add_scalar('loss', losses.item())

            if i % print_freq == 0:
                print(f"Epoch {epoch}, iteration {i}: loss = {losses.item()}")

        # summary.flush()


if __name__ == '__main__':
    # ------------------------------- ARGUMENTS --------------------------------
    parser = argparse.ArgumentParser(description='Task E')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--weights', type=str,
                        default='/ghome/group03/M5-Project/week4/checkpoints/best_loss_task_a_finetunning.h5',
                        help='Path to weights')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    args = parser.parse_args()

    # ------------------------------- PATHS --------------------------------
    env_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(env_path, 'mcv/datasets/COCO')

    output_path = os.path.join(env_path, 'M5-Project/week4/Results/Task_e')


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

    # ------------------------------------ EXTRACT FEATURES ----------------------------------------------
    # ------------------------------- DATASET --------------------------------
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()])
    # #  Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. The images are rescaled to [0.0, 1.0]

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)), # resize image
    #     transforms.Pad((0, 32), fill=0), # pad image to (256, 288)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225])
    # ])

    train_data_dir = os.path.join(dataset_path, 'train2014')
    val_data_dir = os.path.join(dataset_path, 'val2014')

    train_annot_path = os.path.join(dataset_path, 'instances_train2014.json')
    val_annot_path = os.path.join(dataset_path, 'instances_val2014.json')

    annotations = json.load(open(os.path.join(dataset_path, 'mcv_image_retrieval_annotations.json')))

    train_dataset = CocoDetection(train_data_dir, train_annot_path, transform=transform)
    val_dataset = CocoDetection(val_data_dir, val_annot_path, transform=transform)

    # ------------------------------- DATALOADER --------------------------------
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, **kwargs)

    # ------------------------------- MODEL --------------------------------
    # Load the pre-trained Faster R-CNN model

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    # num_classes
    num_classes = len(COCO_CLASSES)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    model.eval()

    # ------------------------------- OPTIMIZER --------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # ------------------------------- TRAIN --------------------------------

    # writer = tensorboard.# summaryWriter(log_dir= output_path + '/logs')
    writer = None

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, writer, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, writer, val_loader, device=device, print_freq=10)

    # Save the fine-tuned model
    torch.save(model.state_dict(), output_path + '/fine_tuned_model.pth')

    # # siamese_train_dataset = TripletCOCODataset(train_dataset, split_name='train')
    # # siamese_test_dataset = TripletCOCODataset(test_dataset, split_name='test')

    # # ------------------------------- DATALOADER --------------------------------
    # kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, **kwargs)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    # siamese_train_loader = DataLoader(siamese_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    # siamese_test_loader = DataLoader(siamese_test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # # ------------------------------- MODEL --------------------------------
    # margin = 1.

    # # Pretrained model from torchvision or from checkpoint
    # if args.pretrained:
    #     embedding_net = EmbeddingNet(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)

    # # else:
    # #     weights_model = torch.load(args.weights)['model_state_dict']
    # #     embedding_net = EmbeddingNet(weights=weights_model).to(device)

    # model = SiameseNet(embedding_net).to(device)

    # #--------------------------------- TRAINING --------------------------------  

    # # Loss function
    # loss_func = losses.ContrastiveLoss().to(device)  # margin

    # # Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # # Early stoppper
    # early_stopper = EarlyStopper(patience=50, min_delta=10)

    # # Learning rate scheduler
    # # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)

    # log_interval = 5

    # trainer.fit(siamese_train_loader, siamese_test_loader, model, loss_func, optimizer, lr_scheduler, args.num_epochs, device, log_interval, output_path)

    # # Plot emmbeddings
    # train_embeddings_cl, train_labels_cl = metrics.extract_embeddings(train_loader, model, device)
    # path = os.path.join(output_path, 'train_embeddings.png')
    # metrics.plot_embeddings(train_embeddings_cl, train_labels_cl, path)

    # val_embeddings_cl, val_labels_cl = metrics.extract_embeddings(test_loader, model, device)
    # path = os.path.join(output_path, 'val_embeddings.png')
    # metrics.plot_embeddings(val_embeddings_cl, val_labels_cl, path)

    # metrics.tsne_features(train_embeddings_cl, train_labels_cl, "train", labels=test_dataset.classes, output_dir="Results/Task_b")
    # metrics.tsne_features(val_embeddings_cl, val_labels_cl, "test", labels=test_dataset.classes, output_dir="Results/Task_b")
