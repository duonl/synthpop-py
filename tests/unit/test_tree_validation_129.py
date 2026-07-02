import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import TransformerMixin

class mock_tree:
    
    def fit(self, X, y):
        pass

    def apply(self,X):
        pass


# A tree is fitted consistently when:
# all leaf nodes are returned when apply is called on the same data used for fitting




def test_tree_is_consistent_returns_true_on_consistent_tree():

