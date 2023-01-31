import numpy as np
import pandas as pd
import re
import os
import torch
from torch.utils.data import Dataset, DataLoader
import transformers

def string_preprocessing(string):

  if type(string) == str:
    string =re.sub("[@#$*&87]", "", string, 0, re.IGNORECASE)
    string = re.sub(r'[^\w\s]', "", string, 0, re.IGNORECASE)
    string =re.sub("[:;]", " ", string, 0, re.IGNORECASE)
    string =re.sub("\n", "", string, 0, re.IGNORECASE)
    string =re.sub("\s\s+", " ", string, 0, re.IGNORECASE)
    string =re.sub("The media could not be loaded", "", string, 0, re.IGNORECASE)
    string = string.strip()

    # if string words less than 3 or Nan then set it to "nan"
    splitted_string = string.split(" ")
    if len(splitted_string) < 3 or type(splitted_string) != list:
      string = 'Nan'
    return string
  
  else:
    return 'Nan'

def preprocess_data(df):

  df.drop(df[['customer_name', 'review_date']], axis=1, inplace=True)

  # converting ratings to labels
  df['customer_rating'] = df['customer_rating'].replace([2.0, 1.0], -1)
  df['customer_rating'] = df['customer_rating'].replace([4.0, 5.0], 1)
  df['customer_rating'] = df['customer_rating'].replace(3.0, 0)

  # preprocessing customer review text
  df['customer_review'] = df['customer_review'].map(string_preprocessing)
  df.drop(df[df['customer_review'] == 'Nan'].index, inplace=True)

  return df

def load_data(dataset_dir):

  dataset_path = "./" + dataset_dir
  file_names = os.listdir(dataset_path)

  # fetching file paths
  file_path = []
  for file in file_names:
    file_path.append(os.path.join(dataset_path, file))
    
  df = pd.concat([pd.read_csv(file) for file in file_path], axis=0)

  # preproceesing the loaded data
  df = preprocess_data(df)
  return df

def encode_text(text, max_encoding_len=128, tokenizer_type = "bert-base-cased"):

  tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_type)
  encoding = tokenizer.encode_plus(
      text,
      max_length = max_encoding_len,
      add_special_tokens = True, # adds special tokens to encoded text
      pad_to_max_length = True,
      return_attention_mask = True, # returns attention mask for the text
      return_token_type_ids = False,
      return_tensors = 'pt' # return encoded string as tensors
    )

  return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()

def create_custom_dataset(text, labels, max_encoding_len = 128, tokenizer_type = "bert-base-cased"):

  tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_type)  
  return CustomDataset(text, labels, max_encoding_len, tokenizer_type)

class CustomDataset(Dataset):

  def __init__(self, reviews, labels, max_encoding_len, tokenizer_type = "bert-base-cased"):
    self.reviews = reviews
    self.labels = labels
    self.tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_type)
    self.max_encoding_len = max_encoding_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, index):
    review = self.reviews[index]
    label = self.labels[index]

    encoding = self.tokenizer.encode_plus(
      review,
      max_length = self.max_encoding_len,
      add_special_tokens = True, # adds special tokens to encoded text
      pad_to_max_length = True,
      return_attention_mask = True, # returns attention mask for the text
      return_token_type_ids = False,
      return_tensors = 'pt' # return encoded string as tensors
    )

    return {
      'review_txt': review,
      'encoding': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'label': torch.tensor(label, dtype=torch.long)
    }




