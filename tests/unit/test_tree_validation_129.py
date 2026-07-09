import pytest
from sklearn.base import TransformerMixin, clone
import numpy as np

from synthpop.methods.tree_utils import _fit_decision_tree_with_reachable_leaves, _check_all_leaf_nodes_are_reached

class mock_tree:

    def __init__(self,tree = None,random_state=None):
        self.tree = tree
        self.random_state = random_state

    def fit(self, X, y):
        self.fit_X = X
        self.fit_y = y
        self.tree_ = self.tree 
        self.random_state_ = self.random_state
        return self

    
    def __sklearn_clone__(self):

        result = mock_tree(tree = None,
                         random_state=self.random_state)
        
        if self.tree is None:
            return result
        else:
            tree = clone(self.tree)
            result.tree = tree
            result.tree.parent = result
        return result

class base_tree:
    def __init__(self, leaf_node_ids=None, n_nodes=1, apply_return_val=None,parent = None):
        self.n_nodes = n_nodes
        self.apply_return_val = apply_return_val
        self.parent = parent #to help make assertions about the correct tree being used.
        self.leaf_node_ids = leaf_node_ids

        self.children_left = np.arange(0,n_nodes,dtype= np.int_)
        self.children_right = self.children_left + 10
        self.children_right[leaf_node_ids] = self.children_left[leaf_node_ids]


    def apply(self,X):
        self.apply_X = X
        return self.apply_return_val
    
    def __sklearn_clone__(self):


        return base_tree(leaf_node_ids=self.leaf_node_ids,
                         n_nodes=self.n_nodes,
                         apply_return_val=self.apply_return_val,)


# A tree is fitted consistently when:
# all leaf nodes are returned when apply is called on the same data used for fitting
# all leaf nodes can be found by checking tree.children_left == tree.children_right



def test_check_all_leaf_nodes_are_reached_returns_true_on_consistent_tree():

    tree = base_tree(n_nodes = 10,leaf_node_ids=[2,3,4],apply_return_val=[2,4,4,3,3,3])
    X_train = np.array([1,2,3])

    result = _check_all_leaf_nodes_are_reached(tree=tree,X_train=X_train)

    assert result
    assert tree.apply_X is X_train


def test_check_all_leaf_nodes_are_reached_returns_false_on_inconsistent_tree():

    # leaf 5 not reached by apply
    tree = base_tree(n_nodes = 10,leaf_node_ids=[2,3,4,5],apply_return_val=[2,4,4,3,3,3])
    X_train = np.array([1,2,3])

    result = _check_all_leaf_nodes_are_reached(tree=tree,X_train=X_train)

    assert not result
    assert tree.apply_X is X_train

def test_fit_decision_tree_consistently_firsttime_consistent(mocker):

    mock_check_all_leaf_nodes_are_reached = mocker.patch("synthpop.methods.tree_utils._check_all_leaf_nodes_are_reached",return_value= True)
    decision_tree = mock_tree()
    decision_tree.tree_ = base_tree()

    X = np.array([1,2,3])
    y = np.array([4,5,6])

    result = _fit_decision_tree_with_reachable_leaves(decision_tree=decision_tree,X=X,y=y)
    
    assert result is decision_tree
    assert np.array_equal(result.fit_X,X)
    assert np.array_equal(result.fit_y,y)

    mock_check_all_leaf_nodes_are_reached.assert_called_once_with(tree=decision_tree.tree_,X_train=X)

def test_fit_decision_tree_consistently_retry_when_inconsistent(mocker):

    mock_check_all_leaf_nodes_are_reached = mocker.patch("synthpop.methods.tree_utils._check_all_leaf_nodes_are_reached",side_effect=[False,False,True])
    new_random_states = [222,333]
    mock_create_instance_seed = mocker.patch("synthpop.reproducibility.RandomStateManager.create_instance_seed", side_effect=new_random_states)
    expected_random_states = [3]+new_random_states
    decision_tree = mock_tree(random_state=3)
    decision_tree.tree = base_tree(parent=decision_tree)


    X = np.array([1,2,3])
    y = np.array([4,5,6])

    result = _fit_decision_tree_with_reachable_leaves(decision_tree=decision_tree,X=X,y=y)

    # Assert that tree_is_consistent has been called with the correct arguments.
    # In this case, that is a tree with the random_state parameter set to the return values of create_instance_seed

    for i in range(len(expected_random_states)):
        kwargs = mock_check_all_leaf_nodes_are_reached.mock_calls[i][2]
        tree = kwargs["tree"].parent

        # The following assertion asserts that:
        # - tree_is_consistent has been called the expected number of times
        # - for each call, the random state has been changed before the tree is fit again.
        # - The first attempt is made without altering the seed.
        # - new instances are used (the call to clone)
        assert tree.random_state_ == expected_random_states[i], "The random_state has not been set"

        # The following asserts that:
        # - The tree passed to tree_is_consistent has been fitted before the call.
        # - The call to fit happened with the correct parameters.
        assert np.array_equal(tree.fit_X,X), "tree has not been fitted again with correct X before passing it to tree_is_consistent"
        assert np.array_equal(tree.fit_y,y), "tree has not been fitted again with correct y  before passing it to tree_is_consistent"


    assert result.random_state_ == expected_random_states[-1], "fit_decision_tree_consistently returned the wrong results"


