import json
import fasttext
import numpy as np

def preprocess_caption(caption):
    # Replace special characters and convert to lowercase
    return caption.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")

def one_hot_encode_caption(model, caption):
    return model.get_sentence_vector(caption)

# Load the FastText model
model = fasttext.load_model("/home/mcv/m5/fasttext_wiki.en.bin")

# Load the JSON data
with open("captions_train2014.json", "r") as file:
    data = json.load(file)

# Extract the captions
captions = [entry["caption"] for entry in data["annotations"]]

# Preprocess and one-hot encode the captions
encoded_captions = [one_hot_encode_caption(model, preprocess_caption(caption)) for caption in captions]

# Save the one-hot encoded captions to a new file
with open("encoded_captions_train2014.npy", "wb") as file:
    np.save(file, np.array(encoded_captions))