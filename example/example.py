import argparse
import time
import functools

import nltk
import json
import pandas as pd

from context_attribution import attributor, tree, context_ops, kvcache, model_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Compute attributions")
    parser.add_argument('--dtype', default="float16", choices=["float16", "bloat16", "float32"], help="Data type to use for model")
    parser.add_argument('--use-cache', action="store_true", help="Use cache for past key values")
    parser.add_argument('--input', type=str, required=True, help="Input json file containing question, context, and prompt template.")

    subparsers = parser.add_subparsers(dest="attribution_method", required=True, help="Attribution method to use")

    loo_parser = subparsers.add_parser("loo", help="Use the leave-one-out attribution method")
    loo_parser.add_argument("--model-name", type=str, required=True, help="HF model name or path to use")

    hierarchical_parser = subparsers.add_parser("hierarchical", help="Use the hierarchical attribution method")
    hierarchical_parser.add_argument("--model-name", type=str, required=True, help="HF model name or path to use")
    hierarchical_parser.add_argument("--keep-paragraphs", type=int, required=True, help="Number of paragraphs to keep after scoring at the paragraph level")

    proxy_parser = subparsers.add_parser("proxy", help="Use the proxy modelling method")
    proxy_parser.add_argument("--proxy-model-name", type=str, required=True, help="HF model name or path to use to perform pruning")
    proxy_parser.add_argument("--target-model-name", type=str, required=True, help="HF model name or path to use to perform rescoring")

    pruning_parser = subparsers.add_parser("pruning", help="Use the pruning attribution method")
    pruning_parser.add_argument("--pruning-model-name", type=str, required=True, help="HF model name or path to use to perform pruning")
    pruning_parser.add_argument("--rescoring-model-name", type=str, required=True, help="HF model name or path to use to perform rescoring")
    pruning_parser.add_argument("--keep-sentences", type=int, required=True, help="Number of sentences to keep after pruning")

    return parser.parse_args()


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper

def create_context_tree(context):
    # Create a tree structure for the context
    root = tree.Tree()
    for paragraph in context.split('\n'):
        paragraph_node = tree.Tree({"separator": "\n\n"})
        for sentence in nltk.sent_tokenize(paragraph):
            sentence_node = tree.Tree({"text": sentence, "separator": " "})
            paragraph_node.children.append(sentence_node)
        root.children.append(paragraph_node)
    return root.to_dict()

def main(args):
    att = attributor.get_attributor(**vars(args))

    in_dict = json.load(open(args.input, 'r'))
    prompt_template = in_dict['prompt_template']

    question = in_dict['question']
    context = in_dict['context']
    context_tree = create_context_tree(context)

    response_ids = model_utils.generate_response(att.model, att.tokenizer, {"context_tree": context_tree, "question": question}, prompt_template)["response_ids"]

    attribution_dict, time_elapsed = timeit(att.run)(question, context_tree, prompt_template, response_ids)
    
    texts = []
    attributions = []
    for child in attribution_dict['children']: 
        for grandchild in child['children']:
            try: attributions.append(grandchild['data']['attribution_score'])
            except: continue
            texts.append(grandchild['data']['text'])
    df = pd.DataFrame(list(zip(texts, attributions)),
                columns =['Sentence', 'Attribution Score'])
    df = df.sort_values('Attribution Score', ascending=False)

    print("Time elapsed: {time_elapsed}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
