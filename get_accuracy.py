from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import re
import json
import gc
import pdb
from datasets import load_dataset
import random
import argparse


gc.collect()  # Collect garbage
torch.cuda.empty_cache()
torch.set_grad_enabled(False)
print("Disabled automatic differentiation")
random.seed(60)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_gsm8k_accuracy(model_name, cache_directory, dataset, batch_size=4, output_file="gsm8k_results.json"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_directory).to(device)

    correct_count = 0
    total_count = len(dataset)
    results = []

    for i in tqdm(range(0, total_count, batch_size)):
        batch = dataset[i:i+batch_size]
        questions = batch['question']
        answers = batch['answer']
        if "Distill-Llama-8B" in model_name:
            prompts = [f"Solve the following math problem step by step.\n {q}" for q in questions]
        else:
            prompts = [f"{q}\nLet's think step by step.\n" for q in questions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=800)
        cot_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answer_prompts = [f"{cot}\nTherefore, the answer (arabic numerals) is" for cot in cot_responses]
        inputs = tokenizer(answer_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        final_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for question, answer, cot_response, final_response in zip(questions, answers, cot_responses, final_responses):
            match = re.search(r'\d+(\.\d+)?', final_response.split("Therefore, the answer (arabic numerals) is")[-1])
            if match:
                predicted_answer = float(match.group())
                correct_answer = float(answer.split()[-1].replace(',', ''))
                is_correct = abs(predicted_answer - correct_answer) < 1e-6
                if is_correct:
                    correct_count += 1
            else:
                predicted_answer = None
                is_correct = False

            results.append({
                "question": question,
                "correct_answer": answer,
                "cot_response": cot_response,
                "final_response": final_response,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct
            })

    accuracy = correct_count / total_count

    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump({
            "model_name": model_name,
            "accuracy": accuracy,
            "total_samples": total_count,
            "results": results
        }, f, indent=2)

    return accuracy


def prepare_dataset(num_examples):
    full_dataset = load_dataset("gsm8k", "main", split="test")
    if num_examples >= len(full_dataset):
        return full_dataset
    else:
        sampled_indices = random.sample(range(len(full_dataset)), num_examples)
        return full_dataset.select(sampled_indices)

def main(model_name, cache_dir):
    dataset = prepare_dataset(num_examples=100)
    accuracy = calculate_gsm8k_accuracy(model_name, cache_dir, dataset, output_file="results/deepseek-gsm8k-results.json")
    print(f"Accuracy on GSM8K dataset: {accuracy:.2%}")

if __name__ == "__main__":
    """
    Arguments:
    --model-name (str): name of the model
    --cache-dir (str): directory to store/load cache of model. If value for argument not passed, uses default cache directory. This is optional.

    Example: 
    python get_accuracy.py --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --cache-dir "../../../../../projects/ziyuyao/models/llama3-8b-deepseek" 
    python get_accuracy.py --model-name meta-llama/Meta-Llama-3-8B --cache-dir "../../../../../projects/ziyuyao/models/llama3-8b-cache"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=False)
    args = parser.parse_args()

    model_name = args.model_name
    cache_dir = args.cache_dir
    main(model_name, cache_dir)
