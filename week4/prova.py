import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

# Define the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weigts='COCO_V1')

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

# Load the image and convert to tensor
img = Image.open('/ghome/group03/mcv/datasets/COCO/train2014/COCO_train2014_000000000009.jpg').convert('RGB')
img_tensor = torchvision.transforms.functional.to_tensor(img)

# Run the model on the image
model.eval()
with torch.no_grad():
    predictions = model([img_tensor])

# Visualize the results
fig, ax = plt.subplots(1)
ax.imshow(img)

for i in range(len(predictions[0]['scores'])):
    score = predictions[0]['scores'][i].item()
    label = COCO_CLASSES[predictions[0]['labels'][i]]
    box = predictions[0]['boxes'][i].tolist()

    if score > 0.5:
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor='r', facecolor='none'))
        ax.text(box[0], box[1], f"{label}: {score:.2f}", fontsize=10, color='r')

plt.show()
