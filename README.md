# self-correction-llm

This repository contains code and resources for experimenting with self-correction techniques in DeepSeek-R1-Distill-Llama-8B. It includes scripts for data generation, attention head patching, and various utilities for analyzing model behavior. Note: This was prepared for [Mats](https://www.matsprogram.org/) application.


## Environment Setup
This project is tested in Python 3.9.9.

To get started, set up the environment:
```
python -m venv env 
source env/bin/activate
pip install -r requirements.txt
```

## Experiment 1
Step 1: Synthesize the dataset 
```
python data_generate.py --model-name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --cache-dir "path-to-cache-dir"
```
Step 2: Get Accuracy for DeepSeek-R1-Distill-Llama-8B model
```
python get_accuracy.py --model-name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --cache-dir "path-to-cache-dir"
```
Step 3: Get Accuracy for DeepSeek-R1-Distill-Llama-8B model
```
python get_accuracy.py --model-name "meta-llama/Meta-Llama-3-8B" --cache-dir "path-to-cache-dir"
```
## Experiment 2 & 3
The experiment 2 and 3 that performs direct logit attribution (DLA) across layers and sub-layers for identifying which layers and sub-layers are responsible for promoting the error-correcting reasoning tokens or repeating-error tokens can be found in `dla-experiment.ipynb`






