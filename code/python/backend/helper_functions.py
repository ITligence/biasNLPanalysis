import pandas as pd
import string
from collections import Counter
from typing import Optional, Any
import os 
import numpy as np

# import necessary functions 
from hugginface_interface import hgTokenizerGetTokens, get_number_of_hidden_layers


def text_model_layers(target_model, hg_gated=False, hg_token=None):
    if hg_token is None:
        hg_token = os.getenv("HUGGINGFACE_TOKEN", "")

    # interface 
    n_layer = get_number_of_hidden_layers(
        target_model,
        logging_level="error",
        hg_gated=hg_gated,
        hg_token=hg_token
    )

    return n_layer

def text_tokenize(texts,
                  model="bert-base-uncased",
                  max_token_to_sentence=4,
                  device="cpu",
                  tokenizer_parallelism=False,
                  model_max_length=None,
                  logging_level="error"):
    # Tokenize the texts using the HuggingFace tokenizer
    # interface 
    tokens = hgTokenizerGetTokens(
        text_strings=texts,
        model=model,
        max_token_to_sentence=max_token_to_sentence,
        device=device,
        tokenizer_parallelism=tokenizer_parallelism,
        model_max_length=model_max_length,
        logging_level=logging_level
    )
    
    # Convert the tokens to a list of dictionaries
    tokens_list = [{'tokens': token} for token in tokens]

    return tokens_list

def unite_columns(dataframe, new_col_name):
    dataframe[new_col_name] = dataframe.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return dataframe

def getUniqueWordsAndFreq(x_characters, hg_tokenizer=None, **kwargs):
    if hg_tokenizer is None:
        # Unite all text variables into one
        x_characters2 = unite_columns(x_characters, "x_characters2")

        # unite all rows in the column into one cell
        x_characters3 = ' '.join(x_characters2['x_characters2'].tolist())

        # Convert to lower case
        x_characters4a = x_characters3.lower()

        # Tokenize into single words
        x_characters4b = [word for word in x_characters4a.split() if word.isalpha() or word.isnumeric()]

        # Create dataframe with single words and frequency
        x_characters5 = pd.DataFrame(Counter(x_characters4b).items(), columns=['Var1', 'Freq']).sort_values(by='Freq', ascending=False)
    else:
        x_characters4b = [text_tokenize(text, model=hg_tokenizer, **kwargs) for text in x_characters]
        flat_list = [item for sublist in x_characters4b for item in sublist]
        x_characters5 = pd.DataFrame(Counter(flat_list).items(), columns=['Var1', 'Freq']).sort_values(by='Freq', ascending=False)

    if len(x_characters5) == 1:
        x_characters5.columns = ['Freq']
        x_characters5.reset_index(inplace=True)
        x_characters5.columns = ['Var1', 'Freq']

    singlewords = pd.DataFrame({
        'words': x_characters5['Var1'].astype(str),
        'n': x_characters5['Freq']
    })

    return singlewords


def text_embedding_aggregation(x, aggregation="min"):
    if aggregation == "min":
        min_vector = x.min(axis=0, skipna=True).values
        return min_vector
    elif aggregation == "max":
        max_vector = x.max(axis=0, skipna=True).values
        return max_vector
    elif aggregation == "mean":
        mean_vector = x.mean(axis=0, skipna=True).values
        return mean_vector
    elif aggregation == "concatenate":
        long_vector = x.values.flatten()
        long_vector = pd.DataFrame([long_vector], columns=[f"Dim{i+1}" for i in range(len(long_vector))])
        
        variable_name = x.columns[0]
        
        if not variable_name == "Dim1":
            variable_name = variable_name.replace("Dim1_", "")
            long_vector.columns = [f"{col}_{variable_name}" for col in long_vector.columns]
        
        return long_vector
    elif aggregation == "normalize":
        sum_vector = x.sum(axis=0, skipna=True).values
        normalized_vector = normalize_vector(sum_vector)
        return normalized_vector

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def layer_aggregation_helper(x, aggregation, return_tokens=False):
    aggregated_layers_saved = []

    # Get unique number of token ids in the variable starting with x['token_id']
    number_of_ids = x['token_id'].unique()

    # Loop over the number of tokens
    for i_token_id in number_of_ids:
        # Select all the layers for each token/token_id
        x1 = x[x['token_id'] == i_token_id]
        # Select only dimensions
        x2 = x1.filter(regex='^Dim')
        # Aggregate the dimensions
        x3 = text_embedding_aggregation(x2, aggregation=aggregation)

        aggregated_layers_saved.append(x3)
    
    aggregated_layers_saved1 = pd.DataFrame(aggregated_layers_saved)

    if return_tokens:
        # Number of layers
        number_of_layers = x['layer_number'].unique()
        n_layers = len(number_of_layers)
        tokens = x['tokens'][:len(x['tokens']) // n_layers]
        tokens_df = pd.DataFrame({'tokens': tokens})
        aggregated_layers_saved1 = pd.concat([tokens_df, aggregated_layers_saved1], axis=1)

    return aggregated_layers_saved1

def text_dim_name(word_embeddings, dim_names=True):
    tokens = None
    word_type = None

    if isinstance(word_embeddings, pd.DataFrame):
        word_embeddings = [word_embeddings]
        x_is_dataframe = True
    else:
        x_is_dataframe = False

    # Remove singlewords_we if it exists
    if 'word_type' in word_embeddings:
        word_type = word_embeddings['word_type']
        del word_embeddings['word_type']

    if 'tokens' in word_embeddings:
        tokens = word_embeddings['tokens']
        del word_embeddings['tokens']

    # Adding dimension names
    if dim_names:
        for i_row in range(len(word_embeddings)):
            word_embeddings[i_row].columns = [
                f"{col}_{i_row}" for col in word_embeddings[i_row].columns
            ]
    else:
        for i_row in range(len(word_embeddings)):
            target_variables_names = word_embeddings[i_row].columns
            variable_names = [name.split('_')[0] for name in target_variables_names]
            word_embeddings[i_row].columns = variable_names

    # Attach word embeddings again
    if word_type is not None:
        word_embeddings.append({'word_type': word_type})

    if tokens is not None:
        word_embeddings.append({'tokens': tokens})

    # Return DataFrame if input was a DataFrame
    if x_is_dataframe:
        return word_embeddings[0]

    return word_embeddings
def get_encoding_change(column):
    # Assuming this function changes the encoding of a given pandas Series
    return column.apply(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else x)

def select_character_v_utf8(x):
    # Ensure `x` is a DataFrame
    if isinstance(x, pd.Series):
        colname_x = x.name if x.name else 'column'
        x = pd.DataFrame({colname_x: x})
    elif not isinstance(x, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame or Series")

    # Select all character variables
    x_characters = x.select_dtypes(include=['object', 'string'])

    # Ensure all variables are UTF-8 coded
    x_characters = x_characters.apply(get_encoding_change)
    
    return x_characters

def generate_placement_vector(raw_layers, texts):
    # Extract column name, if there is one
    column_name = texts.columns[0] if isinstance(texts, pd.DataFrame) else None
    context_tokens = None

    if 'value' in raw_layers['context_tokens']:
        context_tokens = raw_layers['context_tokens']['value']
    elif 'texts' in raw_layers['context_tokens']:
        context_tokens = raw_layers['context_tokens']['texts']
    elif column_name and column_name in raw_layers['context_tokens']:
        context_tokens = raw_layers['context_tokens'][column_name]

    if context_tokens is None:
        raise ValueError("Neither raw_layers['context_tokens']['value'] nor raw_layers['context_tokens']['texts'] found or both are None.")

    dimensions = None

    for token_embedding in context_tokens:
        elements = token_embedding.iloc[0]

        if any(elements.str.contains("na", case=False)) or any(elements.str.contains("NA", case=False)):
            if any(token_embedding['tokens'].str.contains("na", case=False)) or any(token_embedding['tokens'].str.contains("NA", case=False)):
                dimensions = token_embedding.shape

    if dimensions is None:
        raise ValueError("No token embeddings with 'NA' or 'na' found.")

    template_na = pd.DataFrame(np.nan, index=range(dimensions[0]), columns=['tokens'] + [f"Dim{i}" for i in range(1, dimensions[1] - 2)])

    modified_embeddings = []

    for token_embedding in context_tokens:
        elements = token_embedding.iloc[0]

        if (((any(elements.str.contains("na", case=False)) or any(elements.str.contains("NA", case=False))) and token_embedding.shape[0] == 3) or token_embedding.shape[0] == 2):
            if any(token_embedding['tokens'].str.contains("na", case=False)) or any(token_embedding['tokens'].str.contains("NA", case=False)) or len(token_embedding['tokens']) == 2:
                token_embedding.iloc[:, 3:] = np.nan

        modified_embeddings.append(token_embedding)

    if 'value' in raw_layers['context_tokens']:
        raw_layers['context_tokens']['value'] = modified_embeddings
    if 'texts' in raw_layers['context_tokens']:
        raw_layers['context_tokens']['texts'] = modified_embeddings
    if column_name and column_name in raw_layers['context_tokens']:
        raw_layers['context_tokens'][column_name] = modified_embeddings

    return raw_layers