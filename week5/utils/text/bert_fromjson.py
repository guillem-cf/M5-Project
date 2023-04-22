import ujson as json
import os

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

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



# Load the JSON data

env_path = os.path.dirname(os.path.abspath(__file__))
# get path of current file
# dataset_path = '../../../../datasets/COCO'
dataset_path = '/ghome/group03/mcv/datasets/COCO'

train_path = os.path.join(dataset_path, 'encoded_captions_train2014_bert.json')
val_path = os.path.join(dataset_path, 'encoded_captions_val2014_bert.json')
with open(train_path, "r") as file:
    data = json.load(file)
# Extract the captions
output_numpy = []

for annotation in tqdm(data):
    embeddings = annotation['caption']
    
    output_numpy.append(np.array(embeddings))
    
output_numpy = np.array(output_numpy)
# Save the numpy array 
with open(os.path.join(dataset_path,"encoded_captions_train2014_bert.npy"), "wb") as file:
    np.save(file, output_numpy)
    
    

# ------ Validation set ------

with open(val_path, "r") as file:
    data = json.load(file)

# Extract the captions
output_numpy = []

for annotation in tqdm(data):
    
    embedding = annotation['caption']
    
    output_numpy.append(np.array(embedding))
    
output_numpy = np.array(output_numpy)
# Save the numpy array 
with open(os.path.join(dataset_path,"encoded_captions_val2014_bert.npy"), "wb") as file:
    np.save(file, output_numpy)