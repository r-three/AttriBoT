import copy
from typing import Dict, List, Tuple, Optional, Generator


class Tree:
    """
    A simple serializable tree data structure
    """
    def __init__(self, data: Optional[Dict] = None, children: Optional[List['Tree']] = None):
        self.data = {} if data is None else data
        self.children = [] if children is None else children
    
    def is_leaf(self) -> bool: 
        """
        Check if the node is a leaf node
        """
        return len(self.children) == 0

    def to_dict(self) -> Dict:
        """
        Serialize the tree to a dictionary
        """
        return {
            'data': self.data,
            'children': [child.to_dict() for child in self.children]
        }

    def copy(self) -> 'Tree':
        """
        Create a deep copy of the tree
        """
        return Tree(data=copy.deepcopy(self.data), children=[child.copy() for child in self.children])

    @classmethod
    def from_dict(cls, d: Dict) -> 'Tree':
        """
        Deserialize the tree from a dictionary
        """
        data = d.get('data')
        children = d.get('children')
        return cls(data=data, children=None if children is None else [cls.from_dict(child) for child in children])
    
    @classmethod
    def from_tree(cls, t: 'Tree', data: Optional[Dict] = None) -> 'Tree':
        """
        Create tree with same structure as `t` but with `data` in each node
        """
        return cls(data=data, children=[cls.from_tree(child, copy.deepcopy(data)) for child in t.children])

    def __repr__(self) -> str:
        """
        A function for pretty-printing the tree's nesting structure mimicking the output of the `tree` unix command.
        Data values at each node are truncated beyond 80 characters.
        """
        def _repr_aux(node: Optional['Tree'] , prefix: str, last: bool) -> str:
            if node is None:
                return ''
            r = prefix
            truncated_data = str({k: str(v)[:80] + "..." if len(str(v)) > 80 else str(v) for k, v in node.data.items()})
            r += ('└── ' if last else '├── ') + truncated_data + '\n'
            for i, child in enumerate(node.children):
                r += _repr_aux(child, prefix + ('    ' if last else '│   '), i == len(node.children) - 1)
            return r

        return _repr_aux(self, '', True) 

    def __eq__(self, other: 'Tree') -> bool:
        """
        Check if two trees are equal
        """
        return self.data == other.data and self.children == other.children


def traverse(*trees: Tree) -> Generator[Tuple[Tree], None, None]:
    """
    Traverse multiple trees depth-first in parallel
    """
    if not all(isinstance(tree, Tree) for tree in trees):
        raise ValueError("All inputs must be of type Tree")
    
    yield trees

    for children in zip(*[tree.children for tree in trees]):
        yield from traverse(*children)


def get_nodes_at_depth(t: Tree, depth: int) -> List[Tree]:
    """
    Get all nodes in `t` at a target depth
    """
    nodes = []

    def _get_nodes_at_depth(t: Tree, current_depth: int, target_depth: int):
        if current_depth == target_depth:
            nodes.append(t)
            return
        for child in t.children:
            _get_nodes_at_depth(child, current_depth + 1, target_depth)

    _get_nodes_at_depth(t, 0, depth)
    return nodes


def any(tree: Tree, pred: callable) -> bool:
    """
    Check if any node in the tree satisfies the predicate
    """
    for node, in traverse(tree):
        if pred(node):
            return True
    return False
