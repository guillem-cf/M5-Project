import json

import numpy as np
import torch
from transformers import BertTokenizer, BertModel

def preprocess_caption(caption):
    # Replace special characters and convert to lowercase
    return caption.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")

def encode_caption(tokenizer, model, caption):
    inputs = tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Install the transformers library if not installed
# !pip install transformers

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").eval()

# Load the JSON data
with open("captions_train2014.json", "r") as file:
    data = json.load(file)

# Extract the captions
captions = [entry["caption"] for entry in data["annotations"]]

# Preprocess and encode the captions
encoded_captions = [encode_caption(tokenizer, model, preprocess_caption(caption)) for caption in captions]

# Save the encoded captions to a new file
with open("encoded_captions_train2014_bert.npy", "wb") as file:
    np.save(file, np.array(encoded_captions))