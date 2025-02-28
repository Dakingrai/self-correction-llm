import json
import pdb
from datasets import load_dataset
import random
import csv
import re
from transformers import AutoTokenizer

seed = 60
random.seed(seed)

def get_last_sentence(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())  # Split by sentence-ending punctuation
    return sentences[-1] if sentences else text  # Return the last sentence or the full text if no split

def save_file(data, file_name, save_csv=True):
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

def prepare_dataset(num_examples):
    full_dataset = load_dataset("gsm8k", "main", split="test")
    if num_examples >= len(full_dataset):
        return full_dataset
    else:
        sampled_indices = random.sample(range(len(full_dataset)), num_examples)
        return full_dataset.select(sampled_indices)


def main():
    # Tokenization length
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    cache_dir = "../../../projects/ziyuyao/models/llama3-8b-deepseek"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    num_examples = 100  # Specify the number of examples you want to use
    dataset = prepare_dataset(num_examples)
    new_dataset = []
    for example in dataset:
        tmp = {}
        last_sentence = get_last_sentence(example["question"])
        tmp["question"] = example["question"] + " " + last_sentence
        last_question_tokens = tokenizer.tokenize(last_sentence)
        if len(last_question_tokens) < 6:
            continue
        tmp["corrupt_question"] = example["question"] + " " + "Repeat the word: " + (tokenizer.tokenize(last_sentence)[0] + " ")*(len(last_question_tokens)-5)
        tmp["clean_output"] = "Wait"
        tmp["corrupt_output"] = tokenizer.tokenize(last_sentence)[0]
        assert len(tokenizer.tokenize(tmp["corrupt_question"])) == len(tokenizer.tokenize(tmp["question"]))
        new_dataset.append(tmp)

    save_file(new_dataset, "data/intervene_100.json", save_csv=True)
    
if __name__ == "__main__":
    main()