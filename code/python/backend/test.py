import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import torch
pandas2ri.activate()

# Import the 'text' package from R
text_package = importr('text')
dplyr = importr("text")

is_cuda_available = torch.cuda.is_available()

print(f"CUDA available: {is_cuda_available}")

# Set the test sentence
test_sentence = "Create embeddings from this sentence"

embedding = text_package.textEmbed(test_sentence, device = "cpu")

tokens_texts_r = embedding.rx2('tokens').rx2('texts')
tokens_texts_py = pandas2ri.rpy2py(tokens_texts_r[0])

# Extract and convert texts$texts to pandas DataFrame
texts_texts_r = embedding.rx2('texts').rx2('texts')
texts_texts_py = pandas2ri.rpy2py(texts_texts_r)

# Print the DataFrames
#print("Tokens Texts DataFrame:")
#print(tokens_texts_py)

#print("\nTexts Texts DataFrame:")
#print(texts_texts_py)

# Convert to list of DataFrames (if needed)
dataframes_list = [tokens_texts_py, texts_texts_py]

# Print the list of DataFrames
#for df in dataframes_list:
    #print(df)