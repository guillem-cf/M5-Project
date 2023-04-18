import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the COCO annotations JSON file
with open('instances_train2014.json', 'r') as f:
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

# Define BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to generate code using BERT
def generate_code(text):
    inputs = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

# Generate code for each annotation using BERT and store in a new DataFrame column
merged_df['generated_code'] = merged_df['caption'].apply(generate_code)

# Drop unnecessary columns
merged_df.drop(['id_x', 'category_id', 'id_y', 'name', 'supercategory'], axis=1, inplace=True)

# Convert the DataFrame to a JSON format and save it to a file
generated_code_json = merged_df.to_json(orient='records')
with open('generated_code_coco_annotations.json', 'w') as f:
    f.write(generated_code_json)

print("Generated code using BERT for COCO 2014 annotations saved in 'generated_code_coco_annotations.json'")