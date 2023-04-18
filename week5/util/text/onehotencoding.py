import json
import os
import numpy as np
import pandas as pd

# Load the COCO annotations JSON file
env_path = os.path.join(os.path.dirname(__file__))
dataset_path = env_path + '../../datasets/COCO/'
with open(env_path +'instances_train2014.json', 'r') as f:
    data = json.load(f)

# Extract the categories and annotations
categories = data['categories']
annotations = data['annotations']

# Create a DataFrame for annotations
annotations_df = pd.DataFrame(annotations)

# Create a DataFrame for categories
categories_df = pd.DataFrame(categories)

# Merge the DataFrames based on category id
merged_df = annotations_df.merge(categories_df, left_on='category_id', right_on='id')

# Create a one-hot encoding of the category names
one_hot_encoding = pd.get_dummies(merged_df['name'])
one_hot_df = pd.concat([merged_df, one_hot_encoding], axis=1)

# Drop unnecessary columns
one_hot_df.drop(['id_x', 'category_id', 'id_y', 'name', 'supercategory'], axis=1, inplace=True)

# Convert the DataFrame to a JSON format and save it to a file
one_hot_json = one_hot_df.to_json(orient='records')
with open('one_hot_coco_annotations.json', 'w') as f:
    f.write(one_hot_json)

print("One-hot encoding for COCO 2014 annotations saved in 'one_hot_coco_annotations.json'")