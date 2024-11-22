from typing import Dict, Tuple

import transformers
import torch

from context_attribution import context_ops


def load_model_and_tokenizer(model_name: str, dtype: str) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    """
    Load model and tokenizer
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True)

    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def tokenize(tokenizer: "PreTrainedTokenizer", text: str, apply_chat_template: bool = True, return_text: bool = False) -> Dict[str, torch.Tensor]:
    """
    Tokenize text and apply chat template if needed
    """
    if apply_chat_template:
        text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    encoding = tokenizer(text, return_tensors="pt")
    return (encoding, text) if return_text else encoding


def detokenize(tokenizer: "PreTrainedTokenizer", ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
    """
    Detokenizes the token ids.
    """
    return tokenizer.decode(ids.squeeze().tolist(), skip_special_tokens=skip_special_tokens)


def generate_response(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer", example: Dict, prompt_template: str, **generate_kwargs) -> Dict[str, torch.Tensor]:
    """
    Generate response for a given example
    """
    default_generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.1,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "return_dict_in_generate": True,
    }
    for k, v in default_generate_kwargs.items():
        generate_kwargs[k] = v

    context = context_ops.flatten_context(example["context_tree"])
    prompt = prompt_template.format(question=example["question"], context=context)
    inputs = tokenize(tokenizer, prompt).to(model.device)
    output = model.generate(**inputs, **generate_kwargs)
    return {
        "prompt_ids": inputs.input_ids,
        "response_ids": output.sequences[:, inputs.input_ids.size(1):],
    }
