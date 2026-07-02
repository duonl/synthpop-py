import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import TransformerMixin
import numpy as np

from synthpop.methods.tree_utils import tree_is_consistent

class mock_tree:

    def __init__(self, leaf_node_ids, n_nodes, apply_return_val):
        self.n_nodes = n_nodes
        self.apply_return_val = apply_return_val

        self.children_left = np.arange(0,n_nodes,dtype= np.int_)
        self.children_right = self.children_left + 10
        self.children_right[leaf_node_ids] = self.children_left[leaf_node_ids]

    def fit(self, X, y):
        self.fit_X = X
        self.fit_y = y
        return self

    def apply(self,X):
        self.apply_X = X
        return self.apply_return_val


# A tree is fitted consistently when:
# all leaf nodes are returned when apply is called on the same data used for fitting
# all leaf nodes can be found by checking tree.children_left == tree.children_right



def test_tree_is_consistent_returns_true_on_consistent_tree():

    tree = mock_tree(n_nodes = 10,leaf_node_ids=[2,3,4],apply_return_val=[2,4,4,3,3,3])
    X_train = np.array([1,2,3])

    result = tree_is_consistent(tree=tree,X_train=X_train)

    assert result
    assert tree.apply_X is X_train


def test_tree_is_consistent_returns_false_on_inconsistent_tree():

    # leaf 5 not reached by apply
    tree = mock_tree(n_nodes = 10,leaf_node_ids=[2,3,4,5],apply_return_val=[2,4,4,3,3,3])
    X_train = np.array([1,2,3])

    result = tree_is_consistent(tree=tree,X_train=X_train)

    assert not result
    assert tree.apply_X is X_train

