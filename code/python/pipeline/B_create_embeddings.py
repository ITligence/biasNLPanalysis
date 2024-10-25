import torch
import torch.nn.functional as F
import time
from transformers import AutoTokenizer, AutoModel
import np 
import datetime

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# mean pooling - take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # last hidden state of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# function to get embeddings
def get_embeddings(sentences, model, tokenizer, batch_size, embedding_type='sentence'):
    sentence_embeddings = []
    token_embeddings = []
    
    start_time = time.time()
    
    # Determine the maximum sequence length for padding
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    max_seq_length = encoded_input['input_ids'].shape[1]

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_length)
        
        # move the encoded inputs to the GPU
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        if embedding_type in ['sentence', 'both']:
            batch_sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings.append(batch_sentence_embeddings.cpu())  # Move to CPU to save memory
        
        if embedding_type in ['token', 'both']:
            batch_token_embeddings = model_output.last_hidden_state
            token_embeddings.append(batch_token_embeddings.cpu())  # Move to CPU to save memory
    
    end_time = time.time()
    
    # Concatenate embeddings from different batches
    if embedding_type in ['sentence', 'both']:
        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    if embedding_type in ['token', 'both']:
        # Pad token embeddings to the maximum sequence length
        for i in range(len(token_embeddings)):
            pad_size = max_seq_length - token_embeddings[i].shape[1]
            token_embeddings[i] = F.pad(token_embeddings[i], (0, 0, 0, pad_size), 'constant', 0)
        token_embeddings = torch.cat(token_embeddings, dim=0)
    
    time_taken = abs(start_time - end_time)
    print(f"Generated Embeddings. Time Taken: {time_taken} Seconds.")
    
    if embedding_type == 'sentence':
        return sentence_embeddings, "empty"
    elif embedding_type == 'token':
        return "empty", token_embeddings
    elif embedding_type == 'both':
        return sentence_embeddings, token_embeddings

# save embeddings
def save_embeddings(sentence_embeddings, 
                    token_embeddings):
    
    print("Sentence embedding shape", sentence_embeddings.shape)
    print("Token embedding shape", token_embeddings.shape)

    # save embeddings
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    if embedding_type == 'sentence' or embedding_type == 'token':
        save_ds_path = f"/mnt/c/Users/Johan/Documents/ITligence/data/embeddings/{embedding_type}_embedding_{model_name}_{time_string}.npy"         
        np.save(save_ds_path, sentence_embeddings)

    elif embedding_type == 'token':
        save_ds_path = f"/mnt/c/Users/Johan/Documents/ITligence/data/embeddings/{embedding_type}_embedding_{model_name}_{time_string}.npy" 
        np.save(save_ds_path, token_embeddings)
    
    elif embedding_type == 'both':
        sentence_ds_path = f"/mnt/c/Users/Johan/Documents/ITligence/data/embeddings/sentence_embedding_{model_name}_{time_string}.npy"
        token_ds_path = f"/mnt/c/Users/Johan/Documents/ITligence/data/embeddings/token_embedding_{model_name}_{time_string}.npy"  
        np.save(sentence_ds_path, token_embeddings)
        np.save(token_ds_path, token_embeddings)

def main(data_path, 
         model_name, 
         batch_size, 
         embedding_type): 
    
    # load dataset
    data = np.load(data_path)
    
    # sentences to create embeddings from 
    sentences = data["text"].to_list()

    # load model and tokenizer from HuggingFace 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    # get embeddings
    sentence_embeddings, token_embeddings = get_embeddings(sentences = sentences, 
                                                           model = model, 
                                                           tokenizer = tokenizer, 
                                                           batch_size = batch_size, 
                                                           embedding_type = embedding_type)
    
    # save embeddings
    save_embeddings(sentence_embeddings = sentence_embeddings, 
                    token_embeddings = token_embeddings)

if __name__ == "__main__": 
    
    data_path = "/mnt/c/Users/Johan/Documents/ITligence/data/cleaned_data/reduced_ds_2024-10-25_17-30-38.npy"
    model = "bert-base-uncased"
    batch_size = 8
    
    # choose the type of embeddings: 'sentence', 'token', 'both'
    embedding_type = "both"

    main(data_path = data_path, 
         model = model, 
         batch_size = batch_size, 
         embedding_type = embedding_type)