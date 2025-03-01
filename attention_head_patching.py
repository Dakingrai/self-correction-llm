from argparse import ArgumentParser
import pdb
from tqdm import tqdm
import pickle
import random
import torch
from torch.utils.data import DataLoader
from plotly_utils import imshow
import gc
import argparse
from general_utils import load_model, MyDataset, save_file, get_model_architecture, load_data
# from activation_patching import InterveneAttentionHead
from transformer_lens.utils import get_attention_mask


device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu" )


def logits_to_ave_logit_diff(
    logits,
    answer_tokens,
    per_prompt=False,
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    final_logits = logits[:, -1, :].to("cpu")

    answer_logits = final_logits.gather(dim=-1, index=answer_tokens).to("cpu")

    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def ioi_metric(
    logits,
    answer_tokens,
    corrupted_logit_diff,
    clean_logit_diff,
    per_prompt=False,
):
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens, per_prompt)#, input_lengths)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff)

class InterveneAttentionHead:
    def __init__(self, 
                 model, 
                 positions,
                 intervene_heads,
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.positions = positions
        self.intervene_heads = intervene_heads
        self.verbose = False
        self.hooks = []
        self.model.eval()

        def get_hook(layer, head):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                n = int(output.shape[0]/2)
                for each in self.positions:
                    output[:n, each, head, :] = output[n:, each, head, :]
                return output
            return hook
            
        for layer, head in self.intervene_heads:
            self.hooks.append(self.model.blocks[layer].attn.hook_z.register_forward_hook(get_hook(layer, head)))

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()

def attention_head_localization(model, dataloader: DataLoader, all_positions=False, results_path="results/attention_head.png"):
    """
    Function to localize attention head by node ablation
    """
    all_logit_diffs = []
    for clean_question, corrupt_question, correct_idx, incorrect_idx in dataloader:
        answer_tokens = torch.tensor([correct_idx, incorrect_idx]).T
        
        clean_logits, clean_cache = model.run_with_cache(clean_question)
        clean_cache = clean_cache.to("cpu")
        clean_logits = clean_logits.to("cpu")

        corrupted_logits, corrupted_cache = model.run_with_cache(corrupt_question) # for metric
        corrupted_cache = corrupted_cache.to("cpu")
        corrupted_logits = corrupted_logits.to("cpu")
        
        clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
        corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
        
        # assumption: both clean and corrupted prompt in a pair have the same number of tokens
        if all_positions:
            positions = range(len(model.to_str_tokens(clean_question[0])))
        else:
            positions = [-1]
        
        logit_diffs = []
        for layer_n in tqdm(range(model.cfg.n_layers)):
            layer_logit_diffs = []
            for head_n in range(model.cfg.n_heads):
                with torch.no_grad():
                    with InterveneAttentionHead(model, positions=positions, intervene_heads=[(layer_n, head_n)]):
                        logits = model(corrupt_question+clean_question)
                        logits = logits.to("cpu")
                
                # n = int(len(clean_question[0])) # lengths of clean and corrupted inputs
                n = int(len(model.to_str_tokens(clean_question[0])))
                logit_diff = ioi_metric(logits[:n, :, :], answer_tokens, corrupted_logit_diff, clean_logit_diff)
                layer_logit_diffs.append(logit_diff)

                del logits
                gc.collect()
                torch.cuda.empty_cache()
            logit_diffs.append(layer_logit_diffs)

        logit_diffs = torch.tensor(logit_diffs)
        all_logit_diffs.append(logit_diffs)

    average_logit_diffs = sum(all_logit_diffs) / len(all_logit_diffs)
    torch.save(average_logit_diffs, results_path.replace(".png", ".pt"))
    fig = imshow(
            average_logit_diffs,
            labels={"x": "Head", "y": "Layer"},
            title="Attention Head Activation Patching",
            width=600,
            return_fig=True
        )
        
    fig.write_image(f"{results_path}")

def main(model_name, cache_dir, data_path, batch_size):
    model = load_model(model_name, cache_dir=cache_dir)
    data = MyDataset(data_path, batch_size, model)

    dataloader = data.to_dataloader(batch_size)

    attention_head_localization(model, dataloader, all_positions=False, results_path=f"results/attention_head-1.png")


if __name__ == "__main__":
    """
    Arguments:
    --model-name (str): name of the model
    --cache-dir (str): directory to store/load cache of model. If value for argument not passed, uses default cache directory. This is optional.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=False)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    main(args.model_name, args.cache_dir, args.data_path, args.batch_size)