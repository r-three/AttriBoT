# AttriBoT: A **B**ag **o**f **T**ricks for Efficiently Approximating Leave-One-Out Context Attribution

We provide the scripts for our experiments and a few examples for how to run each method. We also provide the generated responses for the three datasets (HotpotQA, SQuAD2.0, QASPER) using Llama 3.1 70B Instruct and Qwen 2.5 72B Instruct.

```bash
pip install -e .
cd context_attribution
```

## Preprocess dataset

```bash
python scripts/preprocess_dataset.py --model meta-llama/Llama-3.1-70B-Instruct --dtype float16 --prompt-template 'Answer the question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {question}' --num-examples 10 --output-path data/datasets/hotpot_qa_l70 hotpot_qa
```

The script (1) loads up the specified dataset, (2) preprocesses the context to form a `context_tree` that encodes the hierarchical structure of the context (e.g., different documents, paragraphs within documents, sentences within paragraphs, etc.), (3) formats the example according to the `--prompt-template` argument and generates a response using the specified model. It then writes out a preprocessed huggingface dataset to the specified output path. Dataset has the following form:
```
Dataset({
    features: ['id', 'question', 'answer', 'context_tree', 'response_ids', 'response', 'prompt_template'],
        num_rows: 1000
        })
```

## Compute Attribution

`scripts/compute_attributions.py` runs attribution with the specified model and attribution method (e.g., LOO, hierarchical, pruning) and writes the results out to the specified output directory.

### Compute attribution with exact LOO error
```bash
python scripts/compute_attributions.py --dataset data/datasets/hotpot_qa_l70 --output-dir data/attributions/l70_nokv --dtype float16 loo --model-name meta-llama/Llama-3.1-70B-Instruct
```

### Compute attribution with key-value caching
```bash
python scripts/compute_attributions.py --dataset data/datasets/hotpot_qa_l70 --output-dir data/attributions/l70_kv --dtype float16 --use-cache loo --model-name meta-llama/Llama-3.1-70B-Instruct
```

### Compute attribution with proxy modelling
```bash
python scripts/compute_attributions.py --dataset data/datasets/hotpot_qa_l70 --output-dir data/attributions/l8_kv --dtype float16 --use-cache loo --model-name meta-llama/Llama-3.1-8B-Instruct
```

### Compute attribution with hierarchical attribution
```bash
python scripts/compute_attributions.py --dataset data/datasets/hotpot_qa_l70 --output-dir data/attributions/l70_hr3_kv --dtype float16 --use-cache hierarchical --model-name meta-llama/Llama-3.1-70B-Instruct --keep-paragraphs 3
```

### Compute attribution with proxy model pruning
```bash
python scripts/compute_attributions.py --dataset data/datasets/hotpot_qa_l70 --output-dir data/attributions/l8_pr3_kv --dtype float16 --use-cache pruning --pruning-model-name meta-llama/Llama-3.1-8B-Instruct --rescoring-model-name meta-llama/Llama-3.1-70B-Instruct --keep-sentences 3
```

## Baselines

### Compute attribution with attention weight
```bash
python scripts/compute_attributions.py --dataset data/datasets/hotpot_qa_l70 --output-dir data/attributions/attw --dtype float16 att --model-name meta-llama/Llama-3.1-70B-Instruct
```

### Compute attribution with sentence similarity
```bash
python scripts/compute_attributions.py --dataset data/datasets/hotpot_qa_l70 --output-dir data/attributions/sim --dtype float16 sim --model-name meta-llama/Llama-3.1-70B-Instruct --sent-model-name sentence-transformers/all-MiniLM-L6-v2
```

### Compute attribution with gradient norm
```bash
python scripts/compute_attributions.py --dataset data/datasets/hotpot_qa_l70 --output-dir data/attributions/gradnorm --dtype float16 gradnorm --model-name meta-llama/Llama-3.1-70B-Instruct
```

## Compute ContextCite Attribution
```bash
python scripts/compute_contextcite_attributions.py --dataset data/datasets/hotpot_qa_l70 --output-dir data/attributions/l70cc16 --dtype float16 --model-name meta-llama/Llama-3.1-70B-Instruct --num-ablations 16
```

The script uses ContextCite (https://github.com/MadryLab/context-cite) to calculate attribution.