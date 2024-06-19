import os
import sys
import time
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from collections import defaultdict
from functools import reduce
import warnings
import numpy as np

# import necessary functions 
from sortingLayers import sortingLayers
from helper_functions import generate_placement_vector, select_character_v_utf8, layer_aggregation_helper, text_embedding_aggregation, getUniqueWordsAndFreq
from hugginface_interface import hgTransformerGetEmbedding

# This returns the non-concatenated embeddings, i.e, embeddings without the $texts element. 
def pyEmbedRawLayers(
    texts,
    model_name="bert-base-uncased",
    layers=-2,
    return_tokens=True,
    word_type_embeddings=False,
    decontextualize=False,
    keep_token_embeddings=True,
    device="cpu",
    tokenizer_parallelism=False,
    model_max_length=None,
    max_token_to_sentence=4,
    hg_gated=False,
    hg_token=os.getenv("HUGGINGFACE_TOKEN", ""),
    logging_level="error",
    sort=True
):
    if decontextualize and not word_type_embeddings:
        raise ValueError(
            "decontextualize = TRUE & word_type_embeddings = FALSE has not been "
            "implemented in textEmbedRawLayers() at this stage. "
            "When using decontextualize = TRUE you need to create the word_type_embeddings. "
            "To create text embeddings without it would take unnecessary time as it would require "
            "sending the same decontextualized words to a transformer multiple times (while getting the same "
            "results over and over). Consider using textEmbed, to get token embeddings as well as text embeddings."
        )

    if isinstance(layers, int):
        layers = [layers]

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if max(layers) > model.config.num_hidden_layers:
        raise ValueError("You are trying to extract layers that do not exist in this model.")

    if layers[0] < 0:
        n = model.config.num_hidden_layers
        layers = [1 + n + layer for layer in layers]
    
    # located in helper_functions 
    texts = pd.DataFrame([texts], columns=['text'])
    data_character_variables = select_character_v_utf8(texts)

    if not decontextualize:
        x = data_character_variables
        sorted_layers_all_variables = {"context_tokens": []}
        for i_variables, text in enumerate(data_character_variables):
            T1_variable = time.time()
            
            # Python file function to HuggingFace
            # interface
            hg_embeddings = hgTransformerGetEmbedding(
                text_strings=[text],
                model=model_name,
                layers=layers,
                return_tokens=return_tokens,
                device=device,
                tokenizer_parallelism=tokenizer_parallelism,
                model_max_length=model_max_length,
                max_token_to_sentence=max_token_to_sentence,
                hg_gated=hg_gated,
                hg_token=hg_token,
                logging_level=logging_level
            )
            
            #see sortingLayers.py
            if sort:
                variable_x = sortingLayers(
                    x=hg_embeddings,
                    layers=layers,
                    return_tokens=return_tokens
                )
            else:
                variable_x = hg_embeddings

            sorted_layers_all_variables["context_tokens"].append(variable_x)
            # Note: Adding name association like names(x)[[i_variables]] is skipped as it's Python list

            T2_variable = time.time()
            variable_time = T2_variable - T1_variable

            version_seq = f"{i_variables + 1}/{len(data_character_variables)}"
            loop_text = f"Completed layers output for variable: {version_seq}, duration: {variable_time:.2f} seconds\n"
            print(loop_text)

    if word_type_embeddings and not decontextualize:
        individual_tokens = {"context_word_type": [], "tokens": []}

        i_we = pd.concat([pd.DataFrame(sorted_layers_all_variables["context_tokens"])])
        print("KEYS")
        print(i_we.keys())
        i_we2 = i_we.groupby(i_we['tokens']).apply(lambda x: x)
        individual_tokens["context_word_type"] = [i_we2.get_group(x) for x in i_we2.groups]

        num_layers = len(layers)

        for i_context, token_id_df in enumerate(individual_tokens["context_word_type"]):
            token_id_variable = token_id_df['token_id']
            num_token = len(token_id_variable) // num_layers
            token_id = np.sort(np.tile(np.arange(1, num_token + 1), num_layers))
            individual_tokens["context_word_type"][i_context]['token_id'] = token_id

        single_words = [df.iloc[0, 0] for df in individual_tokens["context_word_type"]]
        n = [len(df) // num_layers for df in individual_tokens["context_word_type"]]

        individual_tokens["tokens"] = pd.DataFrame({"words": single_words, "n": n})

        print("Completed layers aggregation for word_type_embeddings.")

    if decontextualize:
        individual_tokens = {"decontext": {}}
        data_character_variables1 = pd.concat([pd.DataFrame(text) for text in data_character_variables], axis=1)

        # see helper functions 
        singlewords = getUniqueWordsAndFreq(data_character_variables1[0], hg_tokenizer=model_name)
        list_words = [word for word in singlewords["words"]]

        # interface
        hg_decontexts_embeddings = hgTransformerGetEmbedding(
            text_strings=list_words,
            model=model_name,
            layers=layers,
            return_tokens=return_tokens,
            device=device,
            tokenizer_parallelism=tokenizer_parallelism,
            model_max_length=model_max_length,
            max_token_to_sentence=max_token_to_sentence,
            hg_gated=hg_gated,
            hg_token=hg_token,
            logging_level=logging_level
        )
        
        # see sortingLayers.py
        if sort:
            individual_tokens["decontext"]["word_type"] = sortingLayers(
                x=hg_decontexts_embeddings,
                layers=layers,
                return_tokens=return_tokens
            )
        else:
            individual_tokens["decontext"]["word_type"] = hg_decontexts_embeddings

        individual_tokens["decontext"]["single_words"] = singlewords

        print("Completed layers aggregation for decontexts embeddings.")

    if not decontextualize and not word_type_embeddings:
        word_embeddings_with_layers = sorted_layers_all_variables
    elif not decontextualize and word_type_embeddings and keep_token_embeddings:
        word_embeddings_with_layers = {**sorted_layers_all_variables, **individual_tokens}
    elif not decontextualize and word_type_embeddings and not keep_token_embeddings:
        word_embeddings_with_layers = individual_tokens
    elif decontextualize and word_type_embeddings:
        word_embeddings_with_layers = individual_tokens

    return word_embeddings_with_layers


# aggregates (concatenates) the word embeddings created by textEmbedRawLayers to give the $texts element. 
def pyEmbedLayerAggregation(word_embeddings_layers,
                                 layers="all",
                                 aggregation_from_layers_to_tokens="concatenate",
                                 aggregation_from_tokens_to_texts="mean",
                                 return_tokens=False,
                                 tokens_select=None,
                                 tokens_deselect=None):

    if return_tokens and aggregation_from_tokens_to_texts is not None:
        raise ValueError("return_tokens = TRUE does not work with aggregation_from_tokens_to_texts not being NULL. "
                         "When aggregating tokens to text, it is not possible to have return_token = TRUE. "
                         "To get both token_embeddings and text_embeddings use textEmbed().")

    # If selecting 'all' layers, find out number of layers to help indicate layer index later in code
    if isinstance(layers, str):
        # Get the first embeddings
        x_layer_unique = list(set(col for col in word_embeddings_layers[0].columns if col.startswith("layer_number")))
        # Get which layers
        layers = list(map(int, x_layer_unique))
        # Remove layer 0 because it is the input layer for the word embeddings
        if layers[0] == 0:
            layers = layers[1:]

    # Loop over the list of variables
    selected_layers_aggregated_tibble = defaultdict(list)
    
    for i, x in enumerate(word_embeddings_layers):
        start_time = time.time()

        # Ensure x is a list (to make it work for single word embeddings that are contextualized)
        if isinstance(x, pd.DataFrame):
            x = [x]

        # Get number of unique layers in the variable starting with "layer_number"
        number_of_layers = list(set(x[0]['layer_number']))

        # Check that the right number of levels are selected
        if len(set(layers).difference(number_of_layers)) > 0:
            raise ValueError("You are trying to aggregate layers that were not extracted. "
                             "In textEmbed, the layers option needs to include all the layers used in context_layers.")

        # Select layers in layers-argument selected from the variable starting with layer_number
        selected_layers = [df[df['layer_number'].isin(layers)] for df in x]

        # Select the tokens (e.g., CLS)
        if tokens_select:
            selected_layers = [df[df['tokens'].isin(tokens_select)] for df in selected_layers]

        # Deselect the tokens (e.g., CLS)
        if tokens_deselect:
            selected_layers = [df[~df['tokens'].isin(tokens_deselect)] for df in selected_layers]
            if tokens_deselect == "[CLS]":
                selected_layers = [df.assign(token_id=df['token_id'] - 1) for df in selected_layers]

        # Aggregate across layers
        # layer_aggregation_helper in helper_functions
        selected_layers_aggregated = [layer_aggregation_helper(df, aggregation=aggregation_from_layers_to_tokens, return_tokens=return_tokens)
                                      for df in selected_layers]
    
        if aggregation_from_tokens_to_texts is None:
            selected_layers_aggregated_tibble[i] = selected_layers_aggregated
        else:
            # text_embedding_aggregation in helper_functions
            selected_layers_tokens_aggregated = [text_embedding_aggregation(df, aggregation=aggregation_from_tokens_to_texts)
                                                 for df in selected_layers_aggregated]
            selected_layers_aggregated_tibble[i] = pd.concat(selected_layers_tokens_aggregated, ignore_index=True)

        # Add informative comments (assuming comments are stored in a dictionary or other metadata structure)
        original_comment = word_embeddings_layers.get_comment()
        layers_string = " ".join(map(str, layers))
        comment = (f"{original_comment} textEmbedLayerAggregation: layers = {layers_string} "
                   f"aggregation_from_layers_to_tokens = {aggregation_from_layers_to_tokens} "
                   f"aggregation_from_tokens_to_texts = {aggregation_from_tokens_to_texts} "
                   f"tokens_select = {tokens_select} tokens_deselect = {tokens_deselect}")
        
        selected_layers_aggregated_tibble[i].attrs['comment'] = comment

        # Timing
        end_time = time.time()
        variable_time = end_time - start_time
        version_seq = f"{i+1}/{len(word_embeddings_layers)}"
        loop_text = (f"Completed layers aggregation (variable {version_seq}, duration: {variable_time:.2f} seconds).\n")
        print(loop_text)

    return dict(selected_layers_aggregated_tibble)


def pyEmbed(texts,
               model_name="bert-base-uncased",
               layers=-2,
               dim_name=True,
               aggregation_from_layers_to_tokens="concatenate",
               aggregation_from_tokens_to_texts="mean",
               aggregation_from_tokens_to_word_types=None,
               keep_token_embeddings=True,
               tokens_select=None,
               tokens_deselect=None,
               decontextualize=False,
               model_max_length=None,
               max_token_to_sentence=4,
               tokenizer_parallelism=False,
               device="cpu",
               hg_gated=False,
               hg_token=None,
               logging_level="error",
               **kwargs):

    if isinstance(texts, pd.Series) and texts.isna().sum() > 0:
        warnings.warn("texts contain NA-values.")
    
    start_time = time.time()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    
    if layers < 0:
        layers = list(range(model.config.num_hidden_layers + layers, model.config.num_hidden_layers))
    
    texts = pd.DataFrame([texts], columns=['text'])
    texts = select_character_v_utf8(texts)

    output = []
    data_character_variables = texts.select_dtypes(include=['object', 'string'])

    for col in data_character_variables.columns:
        texts_col = data_character_variables[col].dropna().tolist()
        
        # Get hidden states/layers
        if (aggregation_from_layers_to_tokens is not None or
            aggregation_from_tokens_to_texts is not None or
            decontextualize):
            all_wanted_layers = pyEmbedRawLayers(
                texts=texts_col,
                model_name=model_name,
                layers=layers,
                return_tokens=True,
                word_type_embeddings=True,
                decontextualize=decontextualize,
                device=device,
                tokenizer_parallelism=tokenizer_parallelism,
                model_max_length=model_max_length,
                max_token_to_sentence=max_token_to_sentence,
                hg_gated=hg_gated,
                hg_token=hg_token,
                logging_level=logging_level,
                **kwargs
            )

        if texts.isna().sum() > 0:
            # located in helper_functions
            all_wanted_layers = generate_placement_vector(
                raw_layers=all_wanted_layers,
                texts=texts_col
            )

        output_dict = {}

        if not decontextualize:
            # Get token-level embeddings with aggregated levels
            if aggregation_from_layers_to_tokens is not None and keep_token_embeddings:
                token_embeddings = pyEmbedLayerAggregation(
                    word_embeddings_layers=all_wanted_layers['context_tokens'],
                    layers=layers,
                    aggregation_from_layers_to_tokens=aggregation_from_layers_to_tokens,
                    aggregation_from_tokens_to_texts=None,
                    return_tokens=True,
                    tokens_select=tokens_select,
                    tokens_deselect=tokens_deselect
                )
                output_dict['tokens'] = token_embeddings

            # Get aggregated token layers
            if aggregation_from_tokens_to_texts is not None:
                aggregated_token_embeddings = pyEmbedLayerAggregation(
                    word_embeddings_layers=all_wanted_layers['context_tokens'],
                    layers=layers,
                    aggregation_from_layers_to_tokens=aggregation_from_layers_to_tokens,
                    aggregation_from_tokens_to_texts=aggregation_from_tokens_to_texts,
                    return_tokens=False,
                    tokens_select=tokens_select,
                    tokens_deselect=tokens_deselect
                )
                output_dict['texts'] = aggregated_token_embeddings

        # Aggregate Word Type embeddings
        if aggregation_from_tokens_to_word_types is not None or decontextualize:
            if not decontextualize:
                individual_word_embeddings_layers = all_wanted_layers['context_word_type']
                individual_words = all_wanted_layers['tokens']
            else:
                individual_word_embeddings_layers = all_wanted_layers['decontext']['word_type']
                individual_words = all_wanted_layers['decontext']['single_words']

            if aggregation_from_tokens_to_word_types == "individually":
                original_aggregation_from_tokens_to_texts = aggregation_from_tokens_to_texts
                aggregation_from_tokens_to_texts = None

            individual_word_embeddings = pyEmbedLayerAggregation(
                word_embeddings_layers=individual_word_embeddings_layers,
                layers=layers,
                aggregation_from_layers_to_tokens=aggregation_from_layers_to_tokens,
                aggregation_from_tokens_to_texts=aggregation_from_tokens_to_texts,
                return_tokens=False,
                tokens_select=tokens_select,
                tokens_deselect=tokens_deselect
            )

            if aggregation_from_tokens_to_word_types == "individually":
                aggregation_from_tokens_to_texts = original_aggregation_from_tokens_to_texts

            individual_word_embeddings = pd.concat(individual_word_embeddings, ignore_index=True)

            if aggregation_from_tokens_to_word_types == "individually":
                individual_words = pd.DataFrame(individual_words)
                row_indices = np.repeat(np.arange(len(individual_words)), individual_words['n'])
                individual_words = individual_words.iloc[row_indices].reset_index(drop=True)
                individual_words['id'] = np.arange(len(individual_words))
                individual_words['n'] = 1

            individual_word_embeddings_words = pd.concat([individual_words, individual_word_embeddings], axis=1)
            output_dict['word_types'] = individual_word_embeddings_words

        output.append(output_dict)

    if len(data_character_variables.columns) > 1:
        final_output = {key: [dic[key] for dic in output if key in dic] for key in output[0]}
    else:
        final_output = output[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Duration to embed text: {elapsed_time:.2f} seconds")

    return final_output
