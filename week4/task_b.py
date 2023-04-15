import argparse
import os
import numpy as np
from sklearn.calibration import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
import tqdm
from dataset.siamese_data import SiameseMITDataset
from models.models import SiameseNet, EmbeddingNet
from utils import metrics, trainer, losses
from utils.early_stopper import EarlyStopper
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
)
import umap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

if __name__ == '__main__':
    # ------------------------------- ARGUMENTS --------------------------------
    parser = argparse.ArgumentParser(description='Task B')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--weights', type=str,
                        default='/ghome/group03/M5-Project/week4/Results/Task_b/task_b_siamese.h5',
                        help='Path to weights')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--inference', type=bool, default=True, help='Inference')
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
   
    embedding_net = EmbeddingNet(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
       
    model = SiameseNet(embedding_net).to(device)

    if args.inference:

        weights = torch.load(args.weights)["state_dict"]

        model.load_state_dict(weights)
        model.eval()
      
    # --------------------------------- TRAINING --------------------------------
    if not args.inference:
        # Train model
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

        trainer.fit(siamese_train_loader, siamese_test_loader, model, loss_func, optimizer, lr_scheduler, args.num_epochs,
                device, log_interval, output_path)

    # Plot emmbeddings
    train_embeddings_cl, train_labels_cl = metrics.extract_embeddings(train_loader, model, device)
    path = os.path.join(output_path, 'train_embeddings.png')
    metrics.plot_embeddings(train_embeddings_cl, train_labels_cl, path)

    val_embeddings_cl, val_labels_cl = metrics.extract_embeddings(test_loader, model, device)
    path = os.path.join(output_path, 'val_embeddings.png')
    metrics.plot_embeddings(val_embeddings_cl, val_labels_cl, path)


    # ------------------------------- METRICS and retrieval --------------------------------
    # extract number of classes
    num_classes = len(train_dataset.classes)
    
    # Flatten the embeddings to 1D array
    train_embeddings_cl = train_embeddings_cl.reshape(train_embeddings_cl.shape[0], -1)
    val_embeddings_cl = val_embeddings_cl.reshape(val_embeddings_cl.shape[0], -1)

    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

    # Fit the KNN classifier to the train embeddings and labels

    knn.fit(train_embeddings_cl, train_labels_cl)

    # Predict the labels of the validation  embeddings
    train_preds = knn.predict(train_embeddings_cl)
    val_preds = knn.predict(val_embeddings_cl)
    

    # Calculate accuracy for train and validation data
    train_accuracy = accuracy_score(train_labels_cl, train_preds)
    val_accuracy = accuracy_score(val_labels_cl, val_preds)

    print("Train accuracy: ", train_accuracy)
    print("Validation accuracy: ", val_accuracy)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=200)
    ax.set_title("Precision-Recall curve", size=16)
    print("num_classes: ", num_classes)
    for class_id in range(0, num_classes):
        PrecisionRecallDisplay.from_predictions(
            np.where(val_labels_cl == class_id, 1, 0),
            np.where(val_labels_cl== class_id, val_preds, 1 - val_preds),
            ax=ax,
            name="Class " + str(test_dataset.classes[class_id]),
        )
    plt.savefig("Results/Task_b/PrecisionRecallCurve.png")
    plt.close()


    #get test images from the test dataloader
    test_images = []
    for i, (data, target) in enumerate(test_loader): 
        for j in data:    
            test_images.append(j.permute(1, 2, 0))
        """test_images = torch.cat(test_images, dim=0)
        test_images = test_images.cpu().numpy()
        test_images = np.transpose(test_images, (0, 2, 3, 1))
        test_images = test_images * 255
        test_images = test_images.astype(np.uint8)"""

    #get train images from the train dataloader
    train_images = []
    for i, (data, target) in enumerate(train_loader):
        for j in data:    
            test_images.append(j.permute(1, 2, 0))

        """train_images = torch.cat(train_images, dim=0)
        train_images = train_images.cpu().numpy()
        train_images = np.transpose(train_images, (0, 2, 3, 1))
        train_images = train_images * 255
        train_images = train_images.astype(np.uint8)"""

    neigh_dist, neigh_ind = knn.kneighbors(val_embeddings_cl, n_neighbors=5, return_distance=True)

    metrics.plot_retrieval(
    test_images, train_images,val_labels_cl, train_labels_cl, neigh_ind, neigh_dist, output_dir="Results/Task_a", p=""
    )
    metrics.plot_retrieval(
        test_images, train_images, val_labels_cl, train_labels_cl, neigh_ind, neigh_dist, output_dir="Results/Task_a", p="BEST"
    )
    metrics.plot_retrieval(
        test_images, train_images, val_labels_cl, train_labels_cl, neigh_ind, neigh_dist, output_dir="Results/Task_a", p="WORST"
    )

    # TSNE

    metrics.tsne_features(train_embeddings_cl, train_labels_cl, "train", labels=test_dataset.classes,
                          output_dir="Results/Task_b")
    metrics.tsne_features(val_embeddings_cl, val_labels_cl, "test", labels=test_dataset.classes,
                          output_dir="Results/Task_b")
    
    """# UMAP

    # Load the 2D embeddings and image filenames

    # Fit UMAP to the embeddings
    umap_emb = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean').fit_transform(val_embeddings_cl)

    # Plot the UMAP embeddings
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], alpha=0.5)
    j = 0
    # Retrieve and plot some images based on their UMAP embeddings
    for batch_idx, (data, target) in enumerate(train_loader):
        for i in data:

            plt.imshow(i.permute(1, 2, 0))
            j += 1
            if j == 10:
                break

    # Save the plot
    plt.savefig("Results/Task_b/UMAP.png")"""


            



