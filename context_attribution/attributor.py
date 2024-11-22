from abc import ABC, abstractmethod
from typing import Dict, Optional, List

from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from transformers.cache_utils import DynamicCache, HybridCache
import torch
import numpy as np
from tqdm import tqdm
import sentence_transformers

from context_attribution import context_ops, tree, kvcache, model_utils, baseline_utils


class AttributorBase(ABC):
    def __init__(self, model, tokenizer, use_cache: bool = False, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.use_cache = use_cache
        self.generate_kwargs = {
                "max_new_tokens": 512, 
                "do_sample": True, 
                "temperature": 0.1, 
                "eos_token_id": tokenizer.eos_token_id, 
                "pad_token_id": tokenizer.pad_token_id,
                "return_dict_in_generate": True
        }
   
    def compute_log_likelihood(self, model: "PreTrainedModel", prompt_ids: torch.Tensor, response_ids: torch.Tensor, cache: Optional[kvcache.KVCache] = None, update_cache: bool = False) -> float:
        """
        Computes the likelihood of the response given the prompt.
        """
        with torch.no_grad():
            if cache is not None:
                # Get the cached past_key_values associated with all but the last token in the prompt
                past_key_values, remaining_prompt_ids = cache.get(prompt_ids[:, :-1])
                # Concatenate the uncached prompt tokens with the response tokens
                input_ids = torch.cat((remaining_prompt_ids, prompt_ids[:, -1:], response_ids), dim=1)
                # Mask the uncached prompt tokens and set labels for response tokens
                labels = torch.cat((torch.full_like(remaining_prompt_ids, -100), torch.full_like(prompt_ids[:, -1:], -100), response_ids), dim=1)
                if isinstance(model, Gemma2ForCausalLM):
                    breakpoint()
                    output = model(input_ids=input_ids, labels=labels, past_key_values=past_key_values, use_cache=(cache is not None), )
                    # output_ref = model(input_ids=input_ids, labels=labels, past_key_values=None, use_cache=None)
                else:
                    output = model(input_ids=input_ids, labels=labels, past_key_values=past_key_values, use_cache=(cache is not None))
            else:
                past_key_values = None
                input_ids = torch.cat((prompt_ids, response_ids), dim=1)
                labels = torch.cat((torch.full_like(prompt_ids, -100), response_ids), dim=1)
                output = model(input_ids=input_ids, labels=labels, past_key_values=past_key_values, use_cache=(cache is not None))
            log_likelihood = -(output.loss * response_ids.shape[1])

            if update_cache and cache is not None:
                cache.insert(input_ids, output.past_key_values)

        return log_likelihood.detach().cpu().numpy().item()
    
    @abstractmethod
    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        pass

    
class LOOAttributor(AttributorBase):
    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        if isinstance(self.model, Gemma2ForCausalLM):
            cache = kvcache.KVCache(past_key_values = HybridCache(config=self.model.config, max_batch_size=1, max_cache_len=10000, device="cuda", dtype=torch.bfloat16))
        else:
            cache = kvcache.KVCache() if self.use_cache else None
        
        # Compute likelihood of the response given the full context
        full_context = context_ops.flatten_context(context_tree)
        prompt = prompt_template.format(question=question, context=full_context)
        prompt_ids = model_utils.tokenize(self.tokenizer, prompt)["input_ids"].to(self.model.device)
        response_ids = torch.tensor(response_ids).to(self.model.device)
        full_context_likelihood = self.compute_log_likelihood(self.model, prompt_ids, response_ids, cache=cache, update_cache=True)

        # Note: `depth=2` in `context_ops.generate_masked_context masks` masks at the sentence level
        context_tree = tree.Tree.from_dict(context_tree)
        for context, ablated_subtree in tqdm(context_ops.generate_masked_contexts(context_tree, depth=2), leave=False):
            # Compute likelihood of the response given each partially masked context
            partial_prompt = prompt_template.format(question=question, context=context)
            partial_prompt_ids = model_utils.tokenize(self.tokenizer, partial_prompt)["input_ids"].to(self.model.device)
            partial_context_likelihood = self.compute_log_likelihood(self.model, partial_prompt_ids, response_ids, cache=cache, update_cache=False)
            attribution_score = full_context_likelihood - partial_context_likelihood
            ablated_subtree.data["attribution_score"] = attribution_score
        return context_tree.to_dict()


class HierarchicalAttributor(AttributorBase):
    def __init__(self, *args, keep_paragraphs: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_paragraphs = keep_paragraphs

    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        pass_1_cache = kvcache.KVCache() if self.use_cache else None
        pass_2_cache = kvcache.KVCache() if self.use_cache else None

        # Compute likelihood of the response given the full context
        full_context = context_ops.flatten_context(context_tree)
        prompt = prompt_template.format(question=question, context=full_context)
        prompt_ids = model_utils.tokenize(self.tokenizer, prompt)["input_ids"].to(self.model.device)
        response_ids = torch.tensor(response_ids).to(self.model.device)
        full_context_likelihood = self.compute_log_likelihood(self.model, prompt_ids, response_ids, cache=pass_1_cache, update_cache=True)
 
        # First compute attribution scores at the paragraph level
        context_tree = tree.Tree.from_dict(context_tree)
        for context, ablated_subtree in tqdm(context_ops.generate_masked_contexts(context_tree, depth=1), leave=False):
            partial_prompt = prompt_template.format(question=question, context=context)
            partial_prompt_ids = model_utils.tokenize(self.tokenizer, partial_prompt)["input_ids"].to(self.model.device)
            partial_context_likelihood = self.compute_log_likelihood(self.model, partial_prompt_ids, response_ids, cache=pass_1_cache, update_cache=False)
            attribution_score = full_context_likelihood - partial_context_likelihood
            ablated_subtree.data["attribution_score"] = attribution_score
        
        # Keep only the top `keep_paragraphs` subtrees
        attribution_scores = [node.data["attribution_score"] for node in tree.get_nodes_at_depth(context_tree, 1)]
        if len(attribution_scores) == 0: raise Exception("Empty source")   # Skip empty source examples
        cutoff = np.sort(attribution_scores)[-min(self.keep_paragraphs, len(attribution_scores))]
        pruned_context_tree = context_tree.copy()
        pruned_context_tree.children = [child for child in pruned_context_tree.children if child.data["attribution_score"] >= cutoff]
        
        # Recompute likelihood of the response given the pruned context
        pruned_context = context_ops.flatten_context(pruned_context_tree)
        pruned_prompt = prompt_template.format(question=question, context=pruned_context)
        pruned_prompt_ids = model_utils.tokenize(self.tokenizer, pruned_prompt)["input_ids"].to(self.model.device)
        pruned_context_likelihood = self.compute_log_likelihood(self.model, pruned_prompt_ids, response_ids, cache=pass_2_cache, update_cache=True)

        # Now compute attribution scores at the sentence level on the pruned tree
        sentence_nodes = tree.get_nodes_at_depth(context_tree, 2)
        for context, ablated_subtree in tqdm(context_ops.generate_masked_contexts(pruned_context_tree, depth=2), leave=False):
            partial_prompt = prompt_template.format(question=question, context=context)
            partial_prompt_ids = model_utils.tokenize(self.tokenizer, partial_prompt)["input_ids"].to(self.model.device)
            partial_context_likelihood = self.compute_log_likelihood(self.model, partial_prompt_ids, response_ids, cache=pass_2_cache, update_cache=False)
            attribution_score = pruned_context_likelihood - partial_context_likelihood
            for sentence_node in sentence_nodes:
                if sentence_node == ablated_subtree:
                    sentence_node.data["attribution_score"] = attribution_score
                    break
        return context_tree.to_dict()


class ProxyAttributor(AttributorBase):
    def __init__(self, proxy_model, target_model, proxy_tokenizer, target_tokenizer, use_cache: bool = False, **kwargs):
        self.proxy_model = proxy_model
        self.target_model = target_model
        self.proxy_tokenizer = proxy_tokenizer
        self.target_tokenizer = target_tokenizer
        self.model = self.target_model   # For generating response
        self.tokenizer = self.target_tokenizer   # For generating response
        self.use_cache = use_cache

    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        cache = kvcache.KVCache() if self.use_cache else None

        try_sent = 'AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution'
        if self.proxy_tokenizer.encode(try_sent) != self.target_tokenizer.encode(try_sent):
            # Assume the response_id is passed in with rescoring tokenizer
            # TODO: add an error message if doesn't match
            if response_ids[0][-1] == self.target_tokenizer.eos_token_id:
                response_ids = [self.proxy_tokenizer.encode(self.target_tokenizer.decode(response_ids[0][:-1]), add_special_tokens=False)]
            else:
                response_ids = [self.proxy_tokenizer.encode(self.target_tokenizer.decode(response_ids[0]), add_special_tokens=False)]
            response_ids[0].append(self.proxy_tokenizer.eos_token_id)
        else:
            pass
        
        # Compute likelihood of the response given the full context
        full_context = context_ops.flatten_context(context_tree)
        prompt = prompt_template.format(question=question, context=full_context)
        prompt_ids = model_utils.tokenize(self.tokenizer, prompt)["input_ids"].to(self.proxy_model.device)
        response_ids = torch.tensor(response_ids).to(self.proxy_model.device)
        full_context_likelihood = self.compute_log_likelihood(self.proxy_model, prompt_ids, response_ids, cache=cache, update_cache=True)

        # Note: `depth=2` in `context_ops.generate_masked_context masks` masks at the sentence level
        context_tree = tree.Tree.from_dict(context_tree)
        for context, ablated_subtree in tqdm(context_ops.generate_masked_contexts(context_tree, depth=2), leave=False):
            # Compute likelihood of the response given each partially masked context
            partial_prompt = prompt_template.format(question=question, context=context)
            partial_prompt_ids = model_utils.tokenize(self.tokenizer, partial_prompt)["input_ids"].to(self.proxy_model.device)
            partial_context_likelihood = self.compute_log_likelihood(self.proxy_model, partial_prompt_ids, response_ids, cache=cache, update_cache=False)
            attribution_score = full_context_likelihood - partial_context_likelihood
            ablated_subtree.data["attribution_score"] = attribution_score
        return context_tree.to_dict()


class PruningAttributor(AttributorBase):
    def __init__(self, pruning_model, rescoring_model, pruning_tokenizer, rescoring_tokenizer, keep_sentences: int, use_cache: bool = False, **kwargs):
        self.pruning_model = pruning_model
        self.rescoring_model = rescoring_model
        # Note: Assumes that pruning model and rescoring model have the same tokenizer
        self.pruning_tokenizer = pruning_tokenizer
        self.rescoring_tokenizer = rescoring_tokenizer
        self.model = self.rescoring_model   # For generating response
        self.tokenizer = self.rescoring_tokenizer   # For generating response
        self.keep_sentences = keep_sentences
        self.use_cache = use_cache

    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        pruning_cache = kvcache.KVCache() if self.use_cache else None
        rescoring_cache = kvcache.KVCache() if self.use_cache else None

        try_sent = 'AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution'
        if self.pruning_tokenizer.encode(try_sent) != self.rescoring_tokenizer.encode(try_sent):
            # Assume the response_id is passed in with rescoring tokenizer
            # TODO: add an error message if doesn't match
            rescoring_response_ids = response_ids
            if response_ids[0][-1] == self.rescoring_tokenizer.eos_token_id:
                pruning_response_ids = [self.pruning_tokenizer.encode(self.rescoring_tokenizer.decode(response_ids[0][:-1]), add_special_tokens=False)]
            else:
                pruning_response_ids = [self.pruning_tokenizer.encode(self.rescoring_tokenizer.decode(response_ids[0]), add_special_tokens=False)]
            pruning_response_ids[0].append(self.pruning_tokenizer.eos_token_id)
        else:
            pruning_response_ids = response_ids
            rescoring_response_ids = response_ids

        # Compute likelihood of the response given the full context
        full_context = context_ops.flatten_context(context_tree)
        prompt = prompt_template.format(question=question, context=full_context)
        prompt_ids = model_utils.tokenize(self.pruning_tokenizer, prompt)["input_ids"].to(self.pruning_model.device)
        pruning_response_ids = torch.tensor(pruning_response_ids).to(self.pruning_model.device)
        full_context_likelihood = self.compute_log_likelihood(self.pruning_model, prompt_ids, pruning_response_ids, cache=pruning_cache, update_cache=True)
 
        # Compute likelihood of the response given each partially masked context
        # breakpoint()
        context_tree = tree.Tree.from_dict(context_tree)
        for context, ablated_subtree in tqdm(context_ops.generate_masked_contexts(context_tree, depth=2), leave=False):
            partial_prompt = prompt_template.format(question=question, context=context)
            partial_prompt_ids = model_utils.tokenize(self.pruning_tokenizer, partial_prompt)["input_ids"].to(self.pruning_model.device)
            partial_context_likelihood = self.compute_log_likelihood(self.pruning_model, partial_prompt_ids, pruning_response_ids, cache=pruning_cache, update_cache=False)
            attribution_score = full_context_likelihood - partial_context_likelihood
            ablated_subtree.data["pruning_attribution_score"] = attribution_score

        torch.cuda.empty_cache()

        # Keep only the top `keep_sentences` sentences
        attribution_scores = [node.data["pruning_attribution_score"] for node in tree.get_nodes_at_depth(context_tree, 2)]
        cutoff = np.sort(attribution_scores)[-min(self.keep_sentences, len(attribution_scores))]
        pruned_context_tree = context_tree.copy()
        for paragraph_node in tree.get_nodes_at_depth(pruned_context_tree, 1):
            paragraph_node.children = [child for child in paragraph_node.children if child.data["pruning_attribution_score"] >= cutoff]
        
         # Recompute likelihood of the response given the pruned context with the rescoring model
        pruned_context = context_ops.flatten_context(pruned_context_tree)
        pruned_prompt = prompt_template.format(question=question, context=pruned_context)
        pruned_prompt_ids = model_utils.tokenize(self.rescoring_tokenizer, pruned_prompt)["input_ids"].to(self.rescoring_model.device)
        rescoring_response_ids = torch.tensor(rescoring_response_ids).to(self.rescoring_model.device)
        pruned_context_likelihood = self.compute_log_likelihood(self.rescoring_model, pruned_prompt_ids, rescoring_response_ids, cache=rescoring_cache, update_cache=True)

        # Now compute attribution scores at the sentence level on the pruned tree
        sentence_nodes = tree.get_nodes_at_depth(context_tree, 2)
        for context, ablated_subtree in tqdm(context_ops.generate_masked_contexts(pruned_context_tree, depth=2), leave=False):
            partial_prompt = prompt_template.format(question=question, context=context)
            partial_prompt_ids = model_utils.tokenize(self.rescoring_tokenizer, partial_prompt)["input_ids"].to(self.rescoring_model.device)
            partial_context_likelihood = self.compute_log_likelihood(self.rescoring_model, partial_prompt_ids, rescoring_response_ids, cache=rescoring_cache, update_cache=False)
            attribution_score = pruned_context_likelihood - partial_context_likelihood
            for sentence_node in sentence_nodes:
                if sentence_node == ablated_subtree:
                    sentence_node.data["attribution_score"] = attribution_score
                    break
        return context_tree.to_dict()


class CCAttributor(AttributorBase):
    def __init__(self, *args, num_abl: int, abl_kprob: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_abl = num_abl
        self.abl_kprob = abl_kprob

    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        cache = kvcache.KVCache() if self.use_cache else None

        # Compute likelihood of the response given the full context
        full_context = context_ops.flatten_context(context_tree)
        prompt = prompt_template.format(question=question, context=full_context)
        prompt_ids = model_utils.tokenize(self.tokenizer, prompt)["input_ids"].to(self.model.device)
        response_ids = torch.tensor(response_ids).to(self.model.device)
        full_context_likelihood = self.compute_log_likelihood(self.model, prompt_ids, response_ids, cache=cache, update_cache=True)

        # Note: `depth=2` in `context_ops.generate_masked_context masks` masks at the sentence level
        context_tree = tree.Tree.from_dict(context_tree)
        contexts, mask_arrays = baseline_utils.generate_cc_masked_contexts(context_tree, self.num_abl, self.abl_kprob)
        pcls = []
        for context in tqdm(contexts, leave=False):
            # Compute likelihood of the response given each partially masked context
            partial_prompt = prompt_template.format(question=question, context=context)
            partial_prompt_ids = model_utils.tokenize(self.tokenizer, partial_prompt)["input_ids"].to(self.model.device)
            partial_context_likelihood = self.compute_log_likelihood(self.model, partial_prompt_ids, response_ids, cache=cache, update_cache=False)
            pcls.append(partial_context_likelihood)
            # attribution_score = full_context_likelihood - partial_context_likelihood
            # ablated_subtree.data["attribution_score"] = attribution_score
        solver = baseline_utils.LassoRegression()
        mask_arrays = np.array(mask_arrays)
        pcls = np.array(pcls)
        weight, bias = solver.fit(mask_arrays, pcls, response_ids.shape[1])
        nodes = tree.get_nodes_at_depth(context_tree, 2)
        for ind, node in enumerate(nodes):
            node.data['attribution_score'] = weight[ind]
        return context_tree.to_dict()


class AttentionAccumulator:
    def __init__(self, batch_size: int, sequence_length: int, response_length: int):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.response_length = response_length
        self.response_attention = torch.zeros((batch_size, sequence_length))

    def hook_fn(self, module, input, output):
        attention_weights = output[1]
        avg_attention = attention_weights.mean(dim=1)
        resp_tokens_attention = avg_attention[:, self.sequence_length - self.response_length - 1 : self.sequence_length - 1]
        self.response_attention += resp_tokens_attention.sum(dim=1).cpu()
        return (output[0], None, output[2])


class AttAttributor(AttributorBase):
    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        with torch.no_grad():
            # Compute likelihood of the response given the full context
            full_context = context_ops.flatten_context(context_tree)
            prompt = prompt_template.format(question=question, context=full_context)
            prompt_encoding, prompt_text = model_utils.tokenize(self.tokenizer, prompt, return_text=True)
            prompt_ids = prompt_encoding["input_ids"].to(self.model.device)
            response_ids = torch.tensor(response_ids).to(self.model.device)
            
            past_key_values = None
            input_ids = torch.cat((prompt_ids, response_ids), dim=1)

            # Create AttentionAccumulator
            bs, seq_len = input_ids.shape
            resp_len = response_ids.shape[1]
            attention_accumulator = AttentionAccumulator(bs, seq_len, resp_len)

            # Register hook on all attention layers
            hook_handles = []
            for name, module in self.model.named_modules():
                if isinstance(module, LlamaAttention) or isinstance(module, Qwen2Attention):
                    hook_handle = module.register_forward_hook(attention_accumulator.hook_fn)
                    hook_handles.append(hook_handle)

            output = self.model(input_ids=input_ids, use_cache=False, output_attentions=True)
             
            # De-Register all hooks
            for hook_handle in hook_handles:
                hook_handle.remove()
            
            context_tree = tree.Tree.from_dict(context_tree)
            ignore_prefix = 0
            token_start_indices = np.array([prompt_encoding.token_to_chars(i).start for i in range(1, prompt_encoding.input_ids.shape[1] - 1)])
            nodes = tree.get_nodes_at_depth(context_tree, 2)
            for node in nodes:
                sent = node.data['text']
                start_char = prompt_text.find(sent, ignore_prefix)
                if start_char < 0:
                    raise ValueError(f"Cannot find sentence '{sent}' in prompt")
                end_char = start_char + len(sent)
                start_token = np.where(token_start_indices <= start_char)[0][-1]
                end_token = np.where(token_start_indices > end_char)[0][0]
                node.data["attribution_score"] = attention_accumulator.response_attention[:, start_token:end_token].sum().item()
                ignore_prefix = end_char        # TODO: I think the +1 is not necessary and is causing trouble. 

        return context_tree.to_dict()    


class GradientNormAttributor(AttributorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = None
        self.model.model.embed_tokens.register_forward_hook(self.embedding_hook_fn)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def embedding_hook_fn(self, module, input, output):
        self.embeddings = output
        output.requires_grad_(True)
        return output

    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        full_context = context_ops.flatten_context(context_tree)
        prompt = prompt_template.format(question=question, context=full_context)
        prompt_encoding, prompt_text = model_utils.tokenize(self.tokenizer, prompt, return_text=True)
        # prompt_ids = prompt_encoding["input_ids"].to(self.model.device)
        # response_ids = torch.tensor(response_ids).to(self.model.device)
        prompt_ids = prompt_encoding["input_ids"].to('cpu')
        response_ids = torch.tensor(response_ids).to('cpu')

        input_ids = torch.cat((prompt_ids, response_ids), dim=1)
        labels = torch.cat((torch.full_like(prompt_ids, -100), response_ids), dim=1)

        output = self.model(input_ids=input_ids, labels=labels)
        log_likelihood = -(output.loss * response_ids.shape[1])
        embedding_grads = torch.autograd.grad(log_likelihood, self.embeddings)[0]

        context_tree = tree.Tree.from_dict(context_tree)
        ignore_prefix = 0
        token_start_indices = np.array([prompt_encoding.token_to_chars(i).start for i in range(1, prompt_encoding.input_ids.shape[1] - 1)])
        nodes = tree.get_nodes_at_depth(context_tree, 2)
        for node in nodes:
            sent = node.data['text']
            start_char = prompt_text.find(sent, ignore_prefix)
            if start_char < 0:
                raise ValueError(f"Cannot find sentence '{sent}' in prompt")
            end_char = start_char + len(sent)
            start_token = np.where(token_start_indices <= start_char)[0][-1]
            end_token = np.where(token_start_indices > end_char)[0][0]
            node.data["attribution_score"] = torch.norm(embedding_grads[:, start_token:end_token].reshape(-1)).item()
            ignore_prefix = end_char    # Same remove + 1

        return context_tree.to_dict()    

 
class SimAttributor(AttributorBase):
    def __init__(self, *args, sent_model_name, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_model = sentence_transformers.SentenceTransformer(sent_model_name)

    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        context_tree = tree.Tree.from_dict(context_tree)
        nodes = tree.get_nodes_at_depth(context_tree, 2)
        response = self.tokenizer.decode(response_ids[0][:-1])  # To remove the last <|eot_id|>, which shouldn't matter.
        resp_embed = self.sent_model.encode(response)
        for node in nodes:
            source_embed = self.sent_model.encode(node.data['text'])
            similarity = self.sent_model.similarity(resp_embed, source_embed).item()
            node.data['attribution_score'] = similarity
        return context_tree.to_dict()


def get_attributor(**kwargs):
    """
    Factory function to return the appropriate attributor based on the specified kwargs
    """
    assert kwargs["attribution_method"] in ["loo", "hierarchical", "proxy", "pruning", "cc", "att", "sim", "gradnorm"], f"Invalid attribution method: {kwargs['attribution_method']}"
    if kwargs["attribution_method"] == "loo":
        model, tokenizer = model_utils.load_model_and_tokenizer(kwargs["model_name"], kwargs["dtype"])
        return LOOAttributor(model, tokenizer, **kwargs)
    elif kwargs["attribution_method"] == "hierarchical":
        model, tokenizer = model_utils.load_model_and_tokenizer(kwargs["model_name"], kwargs["dtype"])
        return HierarchicalAttributor(model, tokenizer, **kwargs)
    elif kwargs["attribution_method"] == "pruning":
        pruning_model, pruning_tokenizer = model_utils.load_model_and_tokenizer(kwargs["pruning_model_name"], kwargs["dtype"])
        rescoring_model, rescoring_tokenizer = model_utils.load_model_and_tokenizer(kwargs["rescoring_model_name"], kwargs["dtype"])
        return PruningAttributor(pruning_model, rescoring_model, pruning_tokenizer, rescoring_tokenizer, **kwargs)
    elif kwargs["attribution_method"] == "proxy":
        proxy_model, proxy_tokenizer = model_utils.load_model_and_tokenizer(kwargs["proxy_model_name"], kwargs["dtype"])
        target_model, target_tokenizer = model_utils.load_model_and_tokenizer(kwargs["target_model_name"], kwargs["dtype"])
        return ProxyAttributor(proxy_model, target_model, proxy_tokenizer, target_tokenizer, **kwargs)
    elif kwargs["attribution_method"] == "cc":
        model, tokenizer = model_utils.load_model_and_tokenizer(kwargs["model_name"], kwargs["dtype"])
        return CCAttributor(model, tokenizer, **kwargs)
    elif kwargs["attribution_method"] == "att":
        model, tokenizer = model_utils.load_model_and_tokenizer(kwargs["model_name"], kwargs["dtype"])
        return AttAttributor(model, tokenizer, **kwargs)
    elif kwargs["attribution_method"] == "sim":
        model, tokenizer = model_utils.load_model_and_tokenizer(kwargs["model_name"], kwargs["dtype"])
        return SimAttributor(model, tokenizer, **kwargs)
    elif kwargs["attribution_method"] == "gradnorm":
        model, tokenizer = model_utils.load_model_and_tokenizer(kwargs["model_name"], kwargs["dtype"])
        return GradientNormAttributor(model, tokenizer, **kwargs)
