import os
import pandas as pd
import pdb
import json
import csv

import transformer_lens as lens
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformer_lens import HookedTransformer

def get_model_architecture(model, results_path="results/model_architecture.txt"):
    """
    Function to get the model architecture

    Arguments:
    model (torch.nn.Module): model
    results_path (str): path to save the model architecture
    """
    # get named modules of the model
    with open(results_path, 'w') as f:
        for name, module in model.named_modules():
            f.write("*************"*5 + "\n\n\n")
            f.write(f"{name}\n")
            f.write("-------------------\n")
            f.write(f"{module}\n\n")

    print(f"Model architecture saved in {results_path}!!")
    

def save_file(data, file_name, save_csv=False):
    # json
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

    #csv
    if save_csv:
        csv_file = file_name.replace(".json", ".csv")
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("Input data must be a list of dictionaries.")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
            # Create a CSV writer object
            writer = csv.DictWriter(cf, fieldnames=data[0].keys())
            
            # Write the header row
            writer.writeheader()

            # Write the data rows
            writer.writerows(data)


        print(f"Data saved in {file_name} and {csv_file}!!")
    else:
        print(f"Data saved in {file_name}!!")


def collate_data(xs):
    clean_question, corrupt_question, correct_idx, incorrect_idx = zip(*xs)
    clean_question = list(clean_question)
    corrupt_question = list(corrupt_question)
    # correct_idx and incorrect_idx are tuples containing the clean and
    # corrupted token ids for each respective data example in the batch
    return clean_question, corrupt_question, correct_idx, incorrect_idx

class MyDataset(Dataset):
    def __init__(self, filepath, batch_size, model):
        if filepath.endswith(".csv"):
            self.df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported data file format ({filepath})")
        
        # get only first 10 examples
        self.df = self.df.head(4) # Note: due to time constraint, only 10 examples are used
 
        self.model = model
    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['question'], row["corrupt_question"], self.model.to_single_token(" " +row['clean_output']), self.model.to_single_token(" " +row['corrupt_output'])
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=collate_data)



def load_data(data_path: str):
    """
    Function to load the data

    Arguments:
    data_path (str): path to the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file ({data_path}) not found")
    
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    
    elif data_path.endswith(".json"):
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported data file format ({data_path})")

    return data

def load_model(model_name: str, cache_dir: str):
    """
    Function to load the model

    Arguments:
    model_name (str): name of the model
    cache_dir (str): directory to store/load cache of model. If None then uses the default cache directory.
    """
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if 'Distill-Llama-8B' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B", hf_model=model, tokenizer=tokenizer, device='cuda', dtype='float16')
        model.set_use_split_qkv_input(True)
        model.set_use_hook_mlp_in(True)

    elif 'Llama-3-8B' in model_name:
        model = lens.HookedTransformer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", 
            fold_ln=True, 
            center_unembed=True, 
            center_writing_weights=True,
            cache_dir=cache_dir,
            dtype="bfloat16",
            device="cuda"
        )
        model.set_use_split_qkv_input(True)
        model.set_use_hook_mlp_in(True)
    else:
        raise ValueError(f"Unsupported model ({model_name}) for model loading")

    model.eval()
    return model