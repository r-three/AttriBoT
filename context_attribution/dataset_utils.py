import json

import datasets
import numpy as np
import nltk

from context_attribution.tree import Tree


def load_dataset(dataset: str, **kwargs) -> datasets.Dataset:
    """
    Load dataset by dispatching to the appropriate loader function.
    """
    loaders = {
        "hotpot_qa": load_hotpot_qa,
        "squad": load_squad,
        "qasper": load_qasper
    }
    return loaders[dataset](**kwargs)


def load_hotpot_qa(num_examples: int, **kwargs) -> datasets.Dataset:
    """
    Load the HotpotQA dataset.
    """
    ds = datasets.load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    if num_examples > 0:
        ds = ds.select(range(num_examples))

    def create_context_tree(example):
        supporting_facts = set([(title, sent_id) for title, sent_id in zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"])])
        
        # Create a tree structure for the context
        root = Tree()
        for title, paragraph in zip(example["context"]["title"], example["context"]["sentences"]):
            paragraph_node = Tree({"title": title, "separator": "\n\n"})
            for i, sentence in enumerate(paragraph):
                sentence_node = Tree({"text": sentence, "separator": "", "ground_truth": (title, i) in supporting_facts})
                paragraph_node.children.append(sentence_node)
            root.children.append(paragraph_node)
        
        example["context_tree"] = root.to_dict()
        return example

    ds = ds.map(create_context_tree, remove_columns=["supporting_facts", "context", "level", "type"])
    return ds


def load_squad(raw_data_path: str, num_examples: int, **kwargs) -> datasets.Dataset:
    """
    Load the SQuAD dataset.
    """
    with open(raw_data_path, "r") as f:
        data = json.load(f)["data"]
    
    indices = [(example_index, paragraph_index, question_index) for example_index, example in enumerate(data) for paragraph_index, paragraph in enumerate(example["paragraphs"]) for question_index, question in enumerate(paragraph["qas"]) if not question["is_impossible"] and paragraph_index < 10]
    sampled_indices = [indices[i] for i in np.random.choice(len(indices), num_examples, replace=False)] if num_examples > 0 else indices
    
    ds_dict = {
        "id": [],
        "question": [],
        "answer": [],
        "context_tree": [],
    }

    def create_context_tree(example):
        # Create a tree structure for the context
        root = Tree()
        for paragraph in example["paragraphs"][:10]:
            paragraph_node = Tree({"separator": "\n\n"})
            for sentence in nltk.sent_tokenize(paragraph["context"]):
                sentence_node = Tree({"text": sentence, "separator": " "})
                paragraph_node.children.append(sentence_node)
            root.children.append(paragraph_node)
        return root.to_dict()

    for index in sampled_indices:
        example_index, paragraph_index, question_index = index
        example_dict = data[example_index]
        question_dict = example_dict["paragraphs"][paragraph_index]["qas"][question_index]
        ds_dict["id"].append(question_dict["id"])
        ds_dict["question"].append(question_dict["question"])
        ds_dict["answer"].append([answer["text"] for answer in question_dict["answers"]])
        ds_dict["context_tree"].append(create_context_tree(example_dict))

    return datasets.Dataset.from_dict(ds_dict)


def load_qasper(num_examples: int, **kwargs) -> datasets.Dataset:
    """
    Load the QASPer dataset.
    """
    ds = datasets.load_dataset("allenai/qasper", split="train")
    # Filter out examples with context with more than 60 sentences (these tend to be examples where the paper is parsed weirdly)
    ds = ds.filter(lambda ex: sum([len(paragraph) for paragraph in ex["full_text"]["paragraphs"]]) <= 60)
    indices = [(example_index, question_index) for example_index, example in enumerate(ds) for question_index, question in enumerate(example["qas"]["question"])]
    sampled_indices = [indices[i] for i in np.random.choice(len(indices), num_examples, replace=False)] if num_examples > 0 else indices

    ds_dict = {
            "id": [],
            "question": [],
            "answer": [],
            "context_tree": [],
    }


    def create_context_tree(example):
        # Create a tree structure for the context
        root = Tree()
        for section_name, section in zip(example["full_text"]["section_name"], example["full_text"]["paragraphs"]): 
            section_node = Tree({"header": f"{section_name}\n", "separator": "\n\n"})
            for sentence in section:
                sentence_node = Tree({"text": sentence, "separator": " "})
                section_node.children.append(sentence_node)
            root.children.append(section_node)
        return root.to_dict()
    
    for index in sampled_indices:
        example_index, question_index = index
        example_dict = ds[example_index]
        question = example_dict["qas"]["question"][question_index]
        question_id = example_dict["qas"]["question_id"][question_index]
        answers = [answer_list["highlighted_evidence"] for answer_list in example_dict["qas"]["answers"][question_index]["answer"]]
        ds_dict["id"].append(question_id)
        ds_dict["question"].append(question)
        ds_dict["answer"].append(answers)
        ds_dict["context_tree"].append(create_context_tree(example_dict))

    return datasets.Dataset.from_dict(ds_dict)
