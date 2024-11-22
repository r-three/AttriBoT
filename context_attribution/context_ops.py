from typing import List, Dict, Tuple, Generator, Optional, Union

from context_attribution import tree
from context_attribution.tree import Tree


def create_tree_masks(context_tree: Tree, depth: int) -> List[Tree]:
    """
    Create all possible masks by removing nodes at a certain depth from the context tree
    """
    nodes = tree.get_nodes_at_depth(context_tree, depth)
    masks = []
    for node in nodes:
        mask = Tree.from_tree(context_tree, {"value": True})
        for t_node, mask_node in tree.traverse(context_tree, mask):
            if t_node is node:
                mask_node.data["value"] = False
                mask_node.children = [Tree.from_tree(c, {"value": False}) for c in mask_node.children]
                break
        masks.append(mask)  
    return masks


def apply_mask(context_tree: Tree, mask: Tree) -> Tree:
    """
    Apply a mask to a context tree
    """
    masked_tree = Tree.from_tree(context_tree, {})
    for node, mask_node, masked_node in tree.traverse(context_tree, mask, masked_tree):
        if mask_node.data.get("value", True):
            masked_node.data = node.data
        else:
            masked_node.data["_remove"] = True
    
    def _apply_mask(t: Tree) -> Tree:
        # Get new root of masked tree
        if t.data.get("_remove"):
            if sum([tree.any(child, lambda node: not node.data.get("_remove")) for child in t.children]) > 1:
                raise ValueError("Applying mask results in disconnected trees")
            for child in t.children:
                if tree.any(child, lambda node: not node.data.get("_remove")):
                    return _apply_mask(child)
            # In the case where all children are removed by the mask, just return a Tree representing the empty string
            return Tree({"text": ""})
        else:
            t.children = [_apply_mask(child) for child in t.children if not child.data.get("_remove")]
            return t

    masked_tree =_apply_mask(masked_tree) 
    return masked_tree


def invert_mask(mask: Tree) -> Tree:
    """
    Invert a tree mask
    """
    inverted_mask = Tree.from_tree(mask, {"value": True})
    for node, inverted_node in tree.traverse(mask, inverted_mask):
        inverted_node.data["value"] = not node.data.get("value", True)
    return inverted_mask


def generate_masked_contexts(context_tree: Union[Dict, Tree], depth: int) -> Generator[Tuple[str, str], None, None]:
    """
    Generate all possible masked contexts by removing nodes at a certain depth from the context tree.
    For each masked context, the removed subtree is also yielded by the generator.
    """
    if isinstance(context_tree, dict):
        context_tree = Tree.from_dict(context_tree)

    masks = create_tree_masks(context_tree, depth)
    for mask in masks:
        keep_subtree = apply_mask(context_tree, mask)
        remove_subtree = apply_mask(context_tree, invert_mask(mask))
        context = flatten_context(keep_subtree)
        yield context, remove_subtree


def flatten_context(context: Union[Dict, Tree], mask: Optional[Tree] = None) -> str:
    """
    Flatten a context Tree or dictionary into a string
    """
    if isinstance(context, dict):
        context = Tree.from_dict(context)
    
    masked_context = apply_mask(context, mask) if mask is not None else context
    
    if masked_context.is_leaf():
        text = masked_context.data.get("text", "")
    else:
        text = "".join([flatten_context(child) for child in masked_context.children])
    return f"{context.data.get('header', '')}{text}{context.data.get('separator', '')}"
