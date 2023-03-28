import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from dataset.mit import MITDataset
from torch.utils.data import DataLoader
from utils.metrics import accuracy
from tqdm import tqdm
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import numpy as np

finetuned = True
dataset_path = '../../mcv/datasets/MIT_split'
num_classes = 8

if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
    torch.cuda.amp.GradScaler()
else:
    print("CPU is available")
    device = torch.device("cpu")

if finetuned:
    model = resnet50()
    
    # Replace the last fully-connected layer with a new one that outputs 8 classes
    model.fc = torch.nn.Linear(model.fc.in_features, 8)

    model.load_state_dict(torch.load("Results/Task_a/Task_a_Resnet50_finetuned.pth"))
else:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Replace the last fully-connected layer with a new one that outputs 8 classes
    model.fc = torch.nn.Linear(model.fc.in_features, 8)

model = model.to(device)

transform = transforms.Compose([transforms.ToTensor()])

test_dataset = MITDataset(data_dir=dataset_path, split_name='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

loss_fn = torch.nn.CrossEntropyLoss()

y_true = []
y_pred = []
y_pred_arr = np.zeros((0, num_classes))

model.eval()
with torch.no_grad():
    test_loss = 0
    test_acc = 0
    loop = tqdm(test_loader)
    for idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_loss += loss_fn(outputs, labels)
        test_acc += accuracy(outputs, labels)

        softmax = torch.nn.Softmax(dim=1)
        outputs = softmax(outputs)

        y_true.extend(labels.to("cpu").detach().numpy().flatten().tolist())
        y_pred.extend(np.max(outputs.to("cpu").detach().numpy(), axis=1).flatten().tolist())

        print(outputs.to("cpu").detach().numpy().shape)
        y_pred_arr = np.concatenate((y_pred_arr, outputs.to("cpu").detach().numpy()), axis=0)

    test_loss = float(test_loss / (idx + 1))
    test_acc = float(test_acc / (idx + 1))

    print('Test accuracy:', round(test_acc, 4))
    print('Test loss:', round(test_loss, 4))

y_true = np.asarray(y_true).flatten()
y_pred = np.asarray(y_pred).flatten()


fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=200)
for class_id in range(0, num_classes):
    PrecisionRecallDisplay.from_predictions(np.where(y_true==class_id, 1, 0), np.where(y_true==class_id, y_pred, 1-y_pred), ax=ax, name="Class " + str(test_dataset.classes[class_id]))
plt.savefig("Results/Task_a/PrecisionRecallCurve.png")
plt.close()

# Image retrieval:

model_retrieval = torch.nn.Sequential(*(list(model.children())[:-1]))

# Sort y_pred_arr in descending order for each image
sorted_indices = np.argsort(y_pred_arr, axis=1)[:, ::-1]

# Convert y_true to a binary matrix of shape (n_samples, n_classes)
y_true_binary = np.zeros((y_pred.shape[0], num_classes))
y_true_binary[np.arange(len(y_true)), y_true] = 1

# Compute Prec@1 and Prec@5
prec_at_1 = precision_score(y_true_binary, sorted_indices[:, 0], average='weighted')
prec_at_5 = precision_score(y_true_binary, sorted_indices[:, :5], average='weighted')

print("Prec@1:", prec_at_1)
print("Prec@5:", prec_at_5)