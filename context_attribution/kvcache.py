from typing import Tuple, Optional 

import torch
from torch import Tensor


class KVCache:
    """
    A cache to store past_key_values and retrieve and retrieve past_key_values prefixes.
    """
    def __init__(self, input_ids: Optional[Tensor] = None, past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None):
        self.input_ids = input_ids
        self.past_key_values = past_key_values
    
    def insert(self, input_ids: Tensor, past_key_values: Tuple[Tuple[Tensor, Tensor], ...]):
        """
        Insert the past_key_values for input_ids into the cache.
        """
        self.input_ids = input_ids
        self.past_key_values = past_key_values

    def get(self, input_ids: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """
        Retrieve the cached past_key_values for input_ids.
        Returns the cached past_key_values and remaining input_ids that have no cached values.
        """
        if self.input_ids is None or self.past_key_values is None:
            return None, input_ids

        min_length = min(self.input_ids.size(1), input_ids.size(1))
        comparison = self.input_ids[:, :min_length] == input_ids[:, :min_length]
        differing_indices = torch.nonzero(~comparison, as_tuple=False)

        # If there are no differing indices, return the cached `past_key_values` up to `min_length`
        if differing_indices.size(0) == 0:
            return self._get_past_key_values_prefix(min_length), input_ids[:, min_length:]

        # If there are differing indices, return the cached prefix up to the first differing index
        differing_index = differing_indices[0, 1]
        return self._get_past_key_values_prefix(differing_index), input_ids[:, differing_index:]
    
    def _get_past_key_values_prefix(self, prefix_length: int) -> Tuple[Tuple[Tensor, Tensor], ...]:    
        """
        Return the past_key_values up to the prefix_length.
        """
        return tuple((layer_key[:, :, :prefix_length, :], layer_value[:, :, :prefix_length, :]) 
                     for layer_key, layer_value in self.past_key_values)
