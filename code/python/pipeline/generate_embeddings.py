import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
import os as os
import re 
import csv

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time

path = "/mnt/c/Users/Johan/Documents/ITligence"  # Replace with your desired path
os.chdir(path)

data = pd.read_csv("data/uncleaned_data/JerryWeiAIData/train_orig.csv")

########## GENERATE EMBEDDINGS ############

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # Last hidden state of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
sentences = list(data["text"])

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Set batch size
batch_size = 8  # Adjust as needed

# Tokenize sentences in batches
sentence_embeddings = []
start_time = time.time()
for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i+batch_size]
    encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings.append(batch_embeddings)
end_time = time.time()

# Concatenate embeddings from different batches
sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

time_taken = abs(start_time - end_time)
print(f"Generated Embeddings. Time Taken: {time_taken} Seconds.")
print(sentence_embeddings)
