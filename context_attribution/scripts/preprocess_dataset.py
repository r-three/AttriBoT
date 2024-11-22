import argparse
import random

import numpy as np
import torch
from tqdm import tqdm

from context_attribution import dataset_utils, model_utils


def unescape(text):
    return text.encode('utf-8').decode('unicode_escape')


def parse_args():
    parser = argparse.ArgumentParser(description="Split examples from dataset into sources")
    parser.add_argument("--model", type=str, required=True, help="HF model name to use")
    parser.add_argument('--dtype', default="float16", choices=["float16", "bfloat16", "float32"], help="Data type to use for model")
    parser.add_argument("--prompt-template", type=unescape, required=True, help="Template for LM prompt")
    parser.add_argument('--num-examples', type=int, default=-1, help="Number of examples to compute attributions for")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument("--output-path", type=str, required=True, help="Path to output dataset")

    subparsers = parser.add_subparsers(dest="dataset", required=True)
    hotpot_qa_parser = subparsers.add_parser("hotpot_qa")
    squad_parser = subparsers.add_parser("squad")
    squad_parser.add_argument("--raw-data-path", type=str, required=True, help="Path to raw SQuAD data")
    qasper_parser = subparsers.add_parser("qasper")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_response(model, tokenizer, example, prompt_template):
    output = model_utils.generate_response(model, tokenizer, example, prompt_template)
    return {
        "response_ids": output["response_ids"].tolist(),
        "response": model_utils.detokenize(tokenizer, output["response_ids"], skip_special_tokens=False),
    }


def main(args):
    set_seed(args.seed)
    dataset = dataset_utils.load_dataset(**vars(args))
    model, tokenizer = model_utils.load_model_and_tokenizer(args.model, args.dtype)
    
    responses = {"response_ids": [], "response": [], "prompt_template": []}
    for example in tqdm(dataset):
        output = generate_response(model, tokenizer, example, args.prompt_template)
        responses["response_ids"].append(output["response_ids"])
        responses["response"].append(output["response"])
        responses["prompt_template"].append(args.prompt_template)
    
    dataset = dataset.add_column("response_ids", responses["response_ids"])
    dataset = dataset.add_column("response", responses["response"])
    dataset = dataset.add_column("prompt_template", responses["prompt_template"])
   
    dataset.save_to_disk(args.output_path)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
