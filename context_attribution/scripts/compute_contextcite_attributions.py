from typing import Dict
import argparse
import os
import time
import functools

import datasets
from tqdm import tqdm
import jsonlines

from context_attribution import attributor, tree, context_ops, kvcache, model_utils
from context_cite.context_partitioner import BaseContextPartitioner
from context_cite import ContextCiter


class ContextPartitioner(BaseContextPartitioner):
    def __init__(self, context_tree: Dict):
        self.context_tree = tree.Tree.from_dict(context_tree)
        self.context = context_ops.flatten_context(context_tree)

    @property
    def num_sources(self):
        return len(tree.get_nodes_at_depth(self.context_tree, 2))

    def get_source(self, index):
        mask_tree = tree.Tree.from_tree(self.context_tree, {"value": False})
        tree.get_nodes_at_depth(mask_tree, 2)[index].data["value"] = True
        return context_ops.flatten_context(self.context_tree, mask_tree)

    def split_context(self):
        pass

    def get_context(self, mask):
        if mask is None:
            return self.context
        mask_tree = tree.Tree.from_tree(self.context_tree, {"value": False})
        mask_leaves = tree.get_nodes_at_depth(mask_tree, 2)
        for mask_element, mask_leaf in zip(mask, mask_leaves):
            mask_leaf.data["value"] = mask_element
        for mask_internal_node in tree.get_nodes_at_depth(mask_tree, 1):
            mask_internal_node.data["value"] = tree.any(mask_internal_node, lambda n: n.data.get("value"))
        mask_tree.data["value"] = tree.any(mask_tree, lambda n: n.data.get("value"))
        return context_ops.flatten_context(self.context_tree, mask_tree)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute attributions")
    parser.add_argument("--dataset", type=str, required=True, help="Path to preprocessed dataset to use")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save attributions")
    parser.add_argument('--num-examples', type=int, default=-1, help="Number of examples to compute attributions for")
    parser.add_argument('--start-index', type=int, default=0, help="Number of examples to compute attributions for")
    parser.add_argument('--dtype', default="float16", choices=["float16", "bfloat16", "float32"], help="Data type to use for model")
    parser.add_argument("--model-name", type=str, required=True, help="HF model name to use")
    parser.add_argument("--num-ablations", type=int, required=True, help="Number of ContextCite ablations to perform")
    return parser.parse_args()


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper


def main(args):
    dataset = datasets.load_from_disk(args.dataset)
    model, tokenizer = model_utils.load_model_and_tokenizer(args.model_name, dtype=args.dtype)

    os.makedirs(args.output_dir, exist_ok=True)
    with jsonlines.open(os.path.join(args.output_dir, "attributions.jsonl"), "w", flush=True) as writer:
        for i, example in enumerate(tqdm(dataset)):
            if i < args.start_index:
                continue
            if i == args.num_examples:
                break
            context = context_ops.flatten_context(example["context_tree"])
            question = example["question"]
            prompt_template = example["prompt_template"].replace("{question}", "{query}")
            partitioner = ContextPartitioner(example["context_tree"])
            att = ContextCiter(model, tokenizer, context, question, num_ablations=args.num_ablations, prompt_template=prompt_template, partitioner=partitioner)
            att._cache["output"] = att._get_prompt_ids(return_prompt=True)[1] + example["response"]

            attribution_arr, time_elapsed = timeit(att.get_attributions)()

            attribution_tree = tree.Tree.from_dict(example["context_tree"])
            for leaf, attribution in zip(tree.get_nodes_at_depth(attribution_tree, 2), attribution_arr):
                leaf.data["attribution_score"] = attribution.item()
            attribution_dict = attribution_tree.to_dict()

            writer.write({
                "question": example["question"],
                "answer": example["answer"],
                "response": example["response"],
                "time": time_elapsed,
                "attributions": attribution_dict
            })
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
