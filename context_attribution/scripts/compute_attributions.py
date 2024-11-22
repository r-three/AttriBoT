import argparse
import os
import time
import functools

import datasets
from tqdm import tqdm
import jsonlines

from context_attribution import attributor, tree, context_ops, kvcache, model_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Compute attributions")
    parser.add_argument("--dataset", type=str, required=True, help="Path to preprocessed dataset to use")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save attributions")
    parser.add_argument('--num-examples', type=int, default=-1, help="Number of examples to compute attributions for")
    parser.add_argument('--start-index', type=int, default=0, help="Number of examples to compute attributions for")
    parser.add_argument('--dtype', default="float16", choices=["float16", "bfloat16", "float32"], help="Data type to use for model")
    parser.add_argument('--use-cache', action="store_true", help="Use cache for past key values")
    parser.add_argument('--org-tokenizer', type=str, help="Pass in the original tokenizer if using a new model that doesn't share tokenizer.")

    subparsers = parser.add_subparsers(dest="attribution_method", required=True, help="Attribution method to use")

    loo_parser = subparsers.add_parser("loo", help="Use the leave-one-out attribution method")
    loo_parser.add_argument("--model-name", type=str, required=True, help="HF model name to use")

    hierarchical_parser = subparsers.add_parser("hierarchical", help="Use the hierarchical attribution method")
    hierarchical_parser.add_argument("--model-name", type=str, required=True, help="HF model name to use")
    hierarchical_parser.add_argument("--keep-paragraphs", type=int, required=True, help="Number of paragraphs to keep after scoring at the paragraph level")

    pruning_parser = subparsers.add_parser("pruning", help="Use the pruning attribution method")
    pruning_parser.add_argument("--pruning-model-name", type=str, required=True, help="HF model name to use to perform pruning")
    pruning_parser.add_argument("--rescoring-model-name", type=str, required=True, help="HF model name to use to perform rescoring")
    pruning_parser.add_argument("--keep-sentences", type=int, required=True, help="Number of sentences to keep after pruning")

    cc_parser = subparsers.add_parser("cc", help="Use the ContextCite attribution method")
    cc_parser.add_argument("--model-name", type=str, required=True, help="HF model name to use to perform pruning")
    cc_parser.add_argument("--num-abl", type=int, required=True, help="Number of ablations.")
    cc_parser.add_argument("--abl-kprob", type=float, required=True, help="Probability of sentences kept during ablation.")

    att_parser = subparsers.add_parser("att", help="Use the attention weights attribution method")
    att_parser.add_argument("--model-name", type=str, required=True, help="HF model name to use")

    sim_parser = subparsers.add_parser("sim", help="Use the semantic similarity attribution method")
    sim_parser.add_argument("--model-name", type=str, required=True, help="HF model name to use as a place holder.")
    sim_parser.add_argument("--sent-model-name", type=str, required=True, choices=["sentence-transformers/all-MiniLM-L6-v2"], help="Sentence transformer model name to use")
    
    gradnorm_parser = subparsers.add_parser("gradnorm", help="Use the gradient norm attribution method")
    gradnorm_parser.add_argument("--model-name", type=str, required=True, help="HF model to use")
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

    att = attributor.get_attributor(**vars(args))

    if args.org_tokenizer != None:
        org_tokenizer = model_utils.load_tokenizer(args.org_tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    num_examples = args.num_examples
    with jsonlines.open(os.path.join(args.output_dir, "attributions.jsonl"), "w", flush=True) as writer:
        for i, example in enumerate(tqdm(dataset)):
            if i < args.start_index:
                continue
            if i == num_examples:
                break
            if len(context_ops.flatten_context(example['context_tree'])) > 0:
                if args.org_tokenizer != None:
                    if args.attribution_method == 'pruning':
                        raise Exception("The original tokenizer should be the same one as the rescoring tokenizer.")
                    if example['response_ids'][0][-1] == org_tokenizer.eos_token_id:
                        example['response_ids'][0] = example['response_ids'][0][:-1]
                    example["response_ids"] = [att.tokenizer.encode(org_tokenizer.decode(example['response_ids'][0]), add_special_tokens=False)]
                    example["response_ids"][0].append(att.tokenizer.eos_token_id)
                attribution_dict, time_elapsed = timeit(att.run)(example["question"], example["context_tree"], example["prompt_template"], example["response_ids"])
                writer.write({
                    "question": example["question"],
                    "answer": example["answer"],
                    "response": example["response"],
                    "time": time_elapsed,
                    "attributions": attribution_dict
                })
            else:
                # breakpoint()
                num_examples += 1
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
