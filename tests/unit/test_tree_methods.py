import pandas as pd
import numpy as np
import numpy.typing as npt
import pytest
from sklearn.base import TransformerMixin, BaseEstimator

from synthpop.data_processing.missing_value_handling import BaseMissingValueHandler
from synthpop.methods.cart_synth import _AbstractTreeMethod


class TransformStub(TransformerMixin, BaseEstimator):

    def __init__(self, fit_return_value=None, transform_return_value=None):
        self.transform_return_value = transform_return_value
        self.fit_return_value = fit_return_value

    def fit(self, X, y):
        self.fit_X_ = X
        self.fit_y_ = y
        return self.fit_return_value

    def transform(self, X):
        self.transform_X_ = X
        return self.transform_return_value

    def fit_transform(self, X, y=None, **fit_params):
        self.fit_X_ = X
        self.transform_X_ = X
        self.fit_y_ = y

        return self.transform_return_value
    
class StubMissingHandler(BaseMissingValueHandler):
    
    def __init__(self, prepared_for_fit_result,post_synth_transform_result):
        self.prepared_for_fit_result = prepared_for_fit_result
        self.post_synth_transform_result = post_synth_transform_result
    
    def prepare_data_for_fit(self, X, y):
        self.prepare_data_for_fit_X = X
        self.prepare_data_for_fit_y = y
        return self.prepared_for_fit_result
    
    def post_synth_transform(self, X, y):
        self.post_synth_transform_X = X
        self.post_synth_transform_y = y
        return self.post_synth_transform_result
class StubLeafNodeSampler():
    def __init__(self,sample_from_leaves_return_value):
        self.sample_from_leaves_return_value = sample_from_leaves_return_value
    
    def fit_sampler(self, leaf_ids: npt.ArrayLike, y: npt.ArrayLike):
        self.fit_sampler_leaf_ids = leaf_ids
        self.fit_sampler_y = y
        return self
    
    def sample_from_leaves(self, leaf_ids: npt.ArrayLike) -> np.ndarray:
        self.sample_from_leaves_leaf_ids = leaf_ids
        return self.sample_from_leaves_return_value

class TestTreeMethod(_AbstractTreeMethod):
    def __init__(self, *,
                 encoder = None,
                missing_handling = None,
                tree_sampler = None,
                criterion =None,
                splitter=None,
                max_depth=None,
                min_samples_split=None,
                min_samples_leaf=None,
                min_weight_fraction_leaf=None,
                max_features=None,
                max_leaf_nodes=None,
                random_state=None,
                min_impurity_decrease=None,
                class_weight=None,
                ccp_alpha=0):
        super().__init__(encoder=encoder, missing_handling=missing_handling, tree_sampler=tree_sampler, criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=random_state, min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha)

    def _get_encoder(self):
        return TransformStub()
    
    def _get_missing_handling(self):
        return super()._get_missing_handling()

@pytest.fixture
def encoder():
    return TransformStub()

@pytest.fixture
def missing_handling():
    return StubMissingHandler(None,None)

@pytest.fixture
def leafnode_sampler():
    return StubLeafNodeSampler(None)

@pytest.fixture
def tree_method(encoder,missing_handling,leafnode_sampler):
    return TestTreeMethod(encoder=encoder,missing_handling=missing_handling,tree_sampler=leafnode_sampler)

def test_test(tree_method):
    assert isinstance(tree_method,_AbstractTreeMethod)
