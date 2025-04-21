# AttriBoT: A **B**ag **o**f **T**ricks for Efficiently Approximating Leave-One-Out Context Attribution

This repository contains the official code for the paper: ["AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution"](https://arxiv.org/abs/2411.15102).

## Install

```bash
pip install -e .
```

Installs the package in editable mode.

## Example Usage
See `example/example.py` for an example which runs attribution with the specified model and attribution method (e.g., LOO, hierarchical, pruning) and prints the attribution results as a dataframe.

### Leave-One-Out With Key-Value Cache:

```bash
python example/example.py --input example/aurora.json --dtype float16 --use-cache loo --model-name meta-llama/Llama-3.2-1B-Instruct
```

### Hierarchical Attribution

```bash
python example/example.py --input example/aurora.json --dtype float16 --use-cache hierarchical --model-name meta-llama/Llama-3.2-1B-Instruct --keep-paragraphs 3
```

### Proxy Model Pruning

```bash
python example/example.py --input example/aurora.json --dtype float16 --use-cache proxy --proxy-model-name meta-llama/Llama-3.2-1B-Instruct --target-model-name meta-llama/Llama-3.2-3B-Instruct
```

### Proxy Model Pruning

```bash
python example/example.py --input example/aurora.json --dtype float16 --use-cache pruning --pruning-model-name meta-llama/Llama-3.2-1B-Instruct --rescoring-model-name meta-llama/Llama-3.2-3B-Instruct --keep-sentences 3
```

The code has been tested with Llama 3, Qwen 2, Mistral families.

### Citation

If you find this repo helpful, welcome to cite our work:

```
@inproceedings{
  liu2025attribot,
  title={AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution},
  author={Fengyuan Liu and Nikhil Kandpal and Colin Raffel},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://arxiv.org/abs/2411.15102}
}
```
