from typing import List, Dict, Tuple, Generator, Optional, Union

import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline

from context_attribution import tree
from context_attribution.tree import Tree
from context_attribution.context_ops import create_tree_masks, apply_mask, invert_mask, flatten_context

def create_cc_masks(context_tree: Tree, num_abl: int, abl_kprob: float) -> List[Tree]:
    """
    Create all possible masks by removing nodes at a certain depth from the context tree
    """
    nodes = tree.get_nodes_at_depth(context_tree, 2)
    mask_ex = tree.get_nodes_at_depth(Tree.from_tree(context_tree, {"value": True}), 2)[0]
    masks = []
    mask_arrays = []
    for i in range(num_abl):
        mask_array = np.random.choice([False, True], size=len(nodes), p=[1 - abl_kprob, abl_kprob])
        mask = Tree.from_tree(context_tree, {"value": True})
        ind = 0
        # for t_node, mask_node in tree.traverse(context_tree, mask):
        for mask_node in tree.get_nodes_at_depth(mask, 2):
            if mask_node != mask_ex:
                continue
            if mask_array[ind] == False:
                mask_node.data["value"] = False
                mask_node.children = [Tree.from_tree(c, {"value": False}) for c in mask_node.children]
            ind += 1
        masks.append(mask)
        mask_arrays.append(mask_array)
    return masks, mask_arrays

def generate_cc_masked_contexts(context_tree: Union[Dict, Tree], num_abl, abl_kprob) -> Generator[Tuple[str, str], None, None]:
    """
    Generate all possible masked contexts by removing nodes at a certain depth from the context tree.
    For each masked context, the removed subtree is also yielded by the generator.
    """
    if isinstance(context_tree, dict):
        context_tree = Tree.from_dict(context_tree)

    masks, mask_arrays = create_cc_masks(context_tree, num_abl, abl_kprob)
    contexts = []
    for mask in masks:
        keep_subtree = apply_mask(context_tree, mask)
        # remove_subtree = apply_mask(context_tree, invert_mask(mask))      # To be fixed (only invert depth 2)
        context = flatten_context(keep_subtree)
        contexts.append(context)
    return contexts, mask_arrays       # Not used as an iterator


class BaseSolver(ABC):
    """
    A base solver class.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    @abstractmethod
    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]: ...


class LassoRegression(BaseSolver):
    """
    A LASSO solver using the scikit-learn library.

    Attributes:
        lasso_alpha (float):
            The alpha parameter for the LASSO regression. Defaults to 0.01.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    def __init__(self, lasso_alpha: float = 0.01) -> None:
        self.lasso_alpha = lasso_alpha

    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]:
        X = masks.astype(np.float32)
        Y = outputs / num_output_tokens
        scaler = StandardScaler()
        lasso = Lasso(alpha=self.lasso_alpha, random_state=0, fit_intercept=True)
        # Pipeline is ((X - scaler.mean_) / scaler.scale_) @ lasso.coef_.T + lasso.intercept_
        pipeline = make_pipeline(scaler, lasso)
        pipeline.fit(X, Y)
        # Rescale back to original scale
        weight = lasso.coef_ / scaler.scale_
        bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
        return weight * num_output_tokens, bias * num_output_tokens
