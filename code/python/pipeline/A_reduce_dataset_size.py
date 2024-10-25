import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import os as os
import sys 
import re 
import csv
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer

# optional: convert .txt file to .csv file
def txt_to_csv(input_txt_file, output_csv_file):
    # Read the text file
    with open(input_txt_file, 'r') as txt_file:
        lines = txt_file.readlines()

    # Write to the CSV file
    with open(output_csv_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(['label', 'text'])

        for line in lines:
            line = line.strip()
            # Use regex to extract the label (number at the beginning) and the text
            match = re.match(r'^(\d+)\s+(.*)', line)
            if match:
                label, text = match.groups()
                csv_writer.writerow([label, text])

# if selected, make dataset binary
def binarise_labels(data): 

    # 0 = left
    data["label"] = data["label"].replace([0,1,2,3,4], 0)

    # 1 = right
    data["label"] = data["label"].replace([6,7,8,9,10], 1)

    # Remove rows where label is 5 or NaN
    data = data[data["label"].isin([0, 1])]

    return data

# get dataset preview
def get_dataset_preview(dataset): 

    # print head
    print("############# HEAD ##############")
    print(dataset.head())
    print()

    # print shape 
    print("############# SHAPE #############")
    print(dataset.shape)
    print()

    # print unique labels, na values etc 
    print("######## UNIQUE, NA-VALUES ######")
    print("Columns:")
    print(dataset.columns)
    print("Unique Labels: ")
    print(dataset["label"].unique())
    print("Number of Classes:")
    print(len(dataset["label"].unique()))
    print("Number NAs:")
    print(dataset.isna().sum())
    print("Class Balance:")
    print(round(100 * dataset["label"].value_counts(normalize=True)),)

# select only the most important (diverse) sentences
def reduce_dataset_size(dataset,  
                        goal_size): 
    
    # get number of rows and ensure there are samples to reduce 
    current_size = dataset.shape[0]
    if current_size <= goal_size: 
        print("Goal size exceeds current size. Exiting programme.")
        sys.exit()
    
    # compute number samples to reduce
    samples_to_reduce = current_size - goal_size
    num_classes = len(dataset["label"].unique())
    desired_samples_per_class = int(samples_to_reduce / num_classes)

    # initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # lists to hold the reduced sentences and labels
    reduced_sentences = []
    reduced_labels = []

    for label in dataset['label'].unique():
        # get all sentences and labels for the current class
        class_sentences = dataset[dataset['label'] == label]['text']
        
        # fit and transform the sentences
        tfidf_matrix = vectorizer.fit_transform(class_sentences)
        
        # sum the TF-IDF values for each sentence
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        
        # create a temporary DataFrame to hold sentences and their scores
        temp_df = pd.DataFrame({'text': class_sentences, 'score': sentence_scores})
        
        # sort the sentences by their scores
        temp_df = temp_df.sort_values(by='score', ascending=False)
        
        # select the top N sentences for the current class
        top_sentences = temp_df['text'].head(desired_samples_per_class).tolist()
        
        # append the top sentences and their labels to the reduced lists
        reduced_sentences.extend(top_sentences)
        reduced_labels.extend([label] * len(top_sentences))

    # Create the final reduced DataFrame
    reduced_df = pd.DataFrame({'text': reduced_sentences, 'label': reduced_labels})

    print(f"Reduced dataset shape: {reduced_df.shape}")
    print(f"Reduced labels distribution:\n{reduced_df['label'].value_counts()}")

    return reduced_df

def main(path, 
        data_preview, 
        reduce_size, 
        goal_size,
        file_type, 
        make_binary):
    
    # set path
    os.chdir(path)

    # read dataset depending on file-type
    if file_type == "txt": 
        #dataset = txt_to_csv(PATH TO data_preview)
        "test"
    else: 
        dataset = pd.read_csv("data/uncleaned_data/JerryWeiAIData/train_orig.csv")

    if make_binary: 
        dataset = binarise_labels(dataset)

    
    # preview dataset
    if data_preview: 
        get_dataset_preview(dataset)
    
    # reduce size 
    if reduce_size: 
        red_frame = reduce_dataset_size(dataset, 
                            goal_size)
    
    # save dataset
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_ds_path = f"/mnt/c/Users/Johan/Documents/ITligence/data/cleaned_data/reduced_ds_{goal_size}_{time_string}.npy" 
    
    np.save(save_ds_path, red_frame)
    

if __name__ == "__main__": 

    # make user enter these variables thorugh terminal
    path = "/mnt/c/Users/Johan/Documents/ITligence" 
    data_preview = True 

    reduce_size = True 
    goal_size = 2000

    make_binary = False
    file_type = "csv"

    main(path = path, 
         data_preview = data_preview, 
         reduce_size = reduce_size, 
         goal_size = goal_size, 
         file_type = file_type, 
         make_binary = make_binary)

