import pandas as pd
import numpy as np
import numpy.typing as npt
import pytest
from sklearn import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.tree import BaseDecisionTree

from synthpop.data_processing.missing_value_handling import BaseMissingValueHandler
from synthpop.methods.cart_synth import _AbstractTreeMethod
import copy

# stubs ---------------------------
class TransformStub(TransformerMixin, BaseEstimator):

    def __init__(self, fit_return_value=None, transform_return_value=None):
        self.transform_return_value = transform_return_value
        self.fit_return_value = fit_return_value

    def fit(self, X, y):
        self.fit_X_ = X
        self.fit_y_ = y
        return self

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
    
    def __sklearn_clone__(self):
        return copy.copy(self)

class TestTreeMethod(_AbstractTreeMethod):
    def __init__(self, *,
                 encoder = None,
                missing_handling = None,
                tree_sampler = None,
                apply_result=None,
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
        BaseDecisionTree._fit = self.fit_local
        BaseDecisionTree.apply = self.apply_local
        self.apply_result =apply_result


    def _get_encoder(self):
        return TransformStub()
    
    def fit_local(self, X, y):
        self.fit_X_ = X
        self.fit_y_ = y
        return self
    
    def apply_local(self,X):
        self.apply_X_ = X
        return self.apply_result
    
    def _get_missing_handling(self):
        return super()._get_missing_handling()
#---------------------------------------------------


# Test data ---------------------------------------------------------------------------------------
def get_input_test_data():
    num_int_data = [1,2,5,2,5,3]
    num_float_data = [1.2,2.3,3.4,5.6,7.8,8.9]
    string_data1 = ["a","b","c","d","e","f"]
    string_data2 = ["aa","bc","cc","dD","eE","fF"]

    all_data = [num_int_data,num_float_data,string_data1,string_data2]

    base_cases_numpy = [ ({"num_1":np.array(num_int_data),
                              "cat_1":np.array(string_data1),
                              "cat_2":np.array(string_data2),
                              },target_data) for target_data in [num_float_data,string_data1]]
    return base_cases_numpy + [(pd.DataFrame(d[0]),pd.Series(d[1])) for d in base_cases_numpy]
        # ({"num": np.array(num_int_data)}, num_float_data),#predict floats with ints
        # ({"num": np.array(num_int_data)}, string_data),#predict categorical with ints
        # ({"cat": np.array(string_data)}, string_data),#predict categorical with categorical
        # ]

def get_pure_categorical_input():
    num_int_data = [1,2,5,2,5,3]
    num_float_data = [1.2,2.3,3.4,5.6,7.8,8.9]
    string_data = ["a","b","c","d","e","f"]

    all_data = [num_int_data,num_float_data,string_data]

    base_cases_one_pred = [ ({"pred":np.array(string_data)},target_data)  for target_data in [num_float_data,string_data]]
    return base_cases_one_pred
    

def get_encoder_transform_return_data():
    #The result of the transform of encoding is always a np.array of float32
    num_data = [1.1,2.2,3.3,4.4,5.5,6.6]
    numpy_values_1D = [np.array(num_data),np.array([*num_data[0:3],np.nan,*num_data[4:6]])]

    num_data2=np.array([1.1,2.2,3.3,4.4,5.5,6.6])*1.4

    numpy_data_2D = [ np.vstack([v1,v2]).transpose() for v1 in numpy_values_1D for v2 in [num_data,num_data2]]

    numpy_output = numpy_values_1D+numpy_data_2D
    pandas_output_1D = [pd.DataFrame(data,columns=["single_encoded_column"]) for data in numpy_values_1D]
    pandas_output_2D = [pd.DataFrame(data,columns=["first_encoded_column","second_encoded_column"]) for data in numpy_data_2D]
    return numpy_output + pandas_output_1D + pandas_output_2D #+ [pd.Series(data) for data in numpy_values]

def get_missing_handling_prepare_for_fit_return_data():
    num_float_data = np.array([1.2,2.3,3.4,5.6,7.8,8.9])*3.2
    string_data1 = np.array(["ab","ab","cd","df","ef","fg"])
    string_data2 = np.array(["aba","aba","cda","dfa","efa","fga"])

    base_cases = [ ({"cat_1":string_data1,"num_1":num_float_data,"cat_2":string_data2},target_data) for target_data in [num_float_data,string_data1]]
    return base_cases

# Fixtures ----------------------------------------------------------------------------------------
@pytest.fixture(params=get_encoder_transform_return_data())
def param_transform_result_encoder(request):
    return TransformStub(transform_return_value= request.param)

@pytest.fixture
def encoder():
    return TransformStub(transform_return_value=np.array([1.1,2.2,3.3,4.4,5.5,6.6]))

@pytest.fixture(params=get_missing_handling_prepare_for_fit_return_data())
def param_fit_result_missing_handling(request):
    return StubMissingHandler(prepared_for_fit_result=request.param,post_synth_transform_result=np.array([1.1,2.2,3.3,4.4,5.5,6.6]))

@pytest.fixture
def missing_handling():
    num_float_data = np.array([1.2,2.3,3.4,5.6,7.8,8.9])*3.2
    string_data1 = np.array(["ab","ab","cd","df","ef","fg"])
    string_data2 = np.array(["aba","aba","cda","dfa","efa","fga"])
    return StubMissingHandler(prepared_for_fit_result=({"cat_1":string_data1,"num_1":num_float_data,"cat_2":string_data2},num_float_data),
                              post_synth_transform_result=string_data2)


@pytest.fixture
def leafnode_sampler():
    return StubLeafNodeSampler(None)

@pytest.fixture
def apply_result(missing_handling):
    n = len(missing_handling.prepared_for_fit_result[0])
    apply_result = np.array([i%3 for i in range(n)])
    return apply_result

@pytest.fixture
def tree_method(encoder,missing_handling,leafnode_sampler,apply_result):
    return TestTreeMethod(encoder=encoder,missing_handling=missing_handling,tree_sampler=leafnode_sampler,apply_result=apply_result)

# test fit ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("X,y",get_input_test_data())
def test_fit_trains_encoder(X,y,tree_method,encoder,missing_handling,leafnode_sampler,apply_result):

    tree_method = TestTreeMethod(encoder=encoder,missing_handling=missing_handling,tree_sampler=leafnode_sampler,apply_result=apply_result)
    
    tree_method.fit(X,y)
    assert np.array_equal(tree_method.encoders_["cat_1"].fit_X_,X["cat_1"])
    assert np.array_equal(tree_method.encoders_["cat_1"].fit_y_,y)
    assert np.array_equal(tree_method.encoders_["cat_2"].fit_X_,X["cat_2"])
    assert np.array_equal(tree_method.encoders_["cat_2"].fit_y_,y)

    assert len(tree_method.encoders_) ==2
    assert not ( tree_method.encoders_["cat_1"] is tree_method.encoder)
    assert not ( tree_method.encoders_["cat_2"] is tree_method.encoder)
    assert not ( tree_method.encoders_["cat_2"] is tree_method.encoders_["cat_1"])

@pytest.mark.parametrize("X,y",get_input_test_data())
def test_fit_prepare_data_for_fit_is_called(X,y,tree_method,encoder,missing_handling,leafnode_sampler,apply_result):
    tree_method = TestTreeMethod(encoder=encoder,missing_handling=missing_handling,tree_sampler=leafnode_sampler,apply_result=apply_result)

    tree_method.fit(X,y)

    assert np.array_equal(tree_method.missing_handler_.prepare_data_for_fit_X,X)
    assert np.array_equal(tree_method.missing_handler_.prepare_data_for_fit_y,y)
    assert not (tree_method.missing_handler_ is tree_method.missing_handler)

@pytest.mark.parametrize("X,y",get_input_test_data())
def test_fit_data_is_encoded(X,y,tree_method,encoder,param_fit_result_missing_handling,leafnode_sampler,apply_result):

    tree_method = TestTreeMethod(encoder=encoder,missing_handling=param_fit_result_missing_handling,tree_sampler=leafnode_sampler,apply_result=apply_result)

    tree_method.fit(X,y)

    assert np.array_equal(tree_method.encoders_["cat_1"].transform_X_,tree_method.missing_handler_.prepared_for_fit_result[0]["cat_1"])
    assert np.array_equal(tree_method.encoders_["cat_2"].transform_X_,tree_method.missing_handler_.prepared_for_fit_result[0]["cat_2"])

def get_expected_input_for_tree(encoders,data_1D):

    v1 = encoders["cat_1"].transform_return_value
    v2 = encoders["cat_2"].transform_return_value
    if v1.ndim !=1:
        return np.hstack([ v1,v2,
              np.array([data_1D]).transpose()
              ])

    return np.array([v1,v2,data_1D]).transpose()

@pytest.mark.parametrize("X,y",get_input_test_data())
def test_fit_tree_is_fit(X,y,tree_method,param_transform_result_encoder,param_fit_result_missing_handling,leafnode_sampler,apply_result):

    tree_method = TestTreeMethod(encoder=param_transform_result_encoder,missing_handling=param_fit_result_missing_handling,tree_sampler=leafnode_sampler,apply_result=apply_result)

    tree_method.fit(X,y)

    expected_input_for_tree = get_expected_input_for_tree(tree_method.encoders_,tree_method.missing_handler_.prepared_for_fit_result[0]["num_1"])

    assert np.array_equal(expected_input_for_tree,tree_method.fit_X_,equal_nan=True) #np.array_equal(expected_input_for_tree,tree_method.fit_X_,equal_nan=True)

    assert np.array_equal(tree_method.missing_handler_.prepared_for_fit_result[1],tree_method.fit_y_)

@pytest.mark.parametrize("X,y",get_input_test_data())
def test_fit_tree_is_applied(X,y,tree_method,param_transform_result_encoder,param_fit_result_missing_handling,leafnode_sampler,apply_result):

    tree_method = TestTreeMethod(encoder=param_transform_result_encoder,missing_handling=param_fit_result_missing_handling,tree_sampler=leafnode_sampler,apply_result=apply_result)
    tree_method.fit(X,y)

    expected_input_for_tree = get_expected_input_for_tree(tree_method.encoders_,tree_method.missing_handler_.prepared_for_fit_result[0]["num_1"])
    assert np.array_equal(expected_input_for_tree,tree_method.apply_X_,equal_nan=True)
@pytest.mark.parametrize("X,y",get_input_test_data())
def test_fit_sampler_fit(X,y,tree_method):
    tree_method.fit(X,y)

    assert np.array_equal(tree_method.tree_sampler_.fit_sampler_leaf_ids,tree_method.apply_result)
    assert np.array_equal(tree_method.tree_sampler_.fit_sampler_y,tree_method.missing_handler.prepared_for_fit_result[1])
    assert not (tree_method.tree_sampler is tree_method.tree_sampler_)

# test transform ----------------------------------------------------------------------------------

def make_fitted_tree_method(encoder,missing_handling,leafnode_sampler,apply_result):
    tree_method = TestTreeMethod(encoder=encoder,missing_handling=missing_handling,tree_sampler=leafnode_sampler,apply_result=apply_result)
    tree_method.encoders_ = {"cat_1": clone(encoder),"cat_2":clone(encoder)}
    tree_method.missing_handler_ = clone(missing_handling)
    tree_method.tree_sampler_ = clone(leafnode_sampler)

    return tree_method
@pytest.mark.parametrize("X",[v[0] for v in get_input_test_data()])
def test_transform_encodes_data(X,encoder,missing_handling,leafnode_sampler,apply_result):
    tree_method = make_fitted_tree_method(encoder,missing_handling=missing_handling,leafnode_sampler=leafnode_sampler,apply_result=apply_result)

    tree_method.transform(X)

    assert np.array_equal(X["cat_1"],tree_method.encoders_["cat_1"].transform_X_)
    assert np.array_equal(X["cat_2"],tree_method.encoders_["cat_2"].transform_X_)

@pytest.mark.parametrize("X",[v[0] for v in get_input_test_data()])
def test_transform_applies(X,param_transform_result_encoder,missing_handling,leafnode_sampler,apply_result):
    tree_method = make_fitted_tree_method(param_transform_result_encoder,missing_handling=missing_handling,leafnode_sampler=leafnode_sampler,apply_result=apply_result)

    tree_method.transform(X)

    expected_input_for_tree = get_expected_input_for_tree(tree_method.encoders_,X["num_1"])

    assert np.array_equal(expected_input_for_tree,tree_method.apply_X_,equal_nan=True)

@pytest.mark.parametrize("X",[v[0] for v in get_input_test_data()])
def test_transform_samples(X,encoder,missing_handling,leafnode_sampler,apply_result):

    tree_method = make_fitted_tree_method(encoder,missing_handling=missing_handling,leafnode_sampler=leafnode_sampler,apply_result=apply_result)

    tree_method.transform(X)

    assert np.array_equal(apply_result,tree_method.tree_sampler_.sample_from_leaves_leaf_ids)

@pytest.mark.parametrize("X",[v[0] for v in get_input_test_data()])
def test_transform_calls_post_synth_transform(X,encoder,missing_handling,leafnode_sampler,apply_result):
    
    tree_method = make_fitted_tree_method(encoder,missing_handling=missing_handling,leafnode_sampler=leafnode_sampler,apply_result=apply_result)

    result = tree_method.transform(X)

    if isinstance(X,pd.DataFrame):
        assert X.equals(tree_method.missing_handler_.post_synth_transform_X)
    else:
        assert X == tree_method.missing_handler_.post_synth_transform_X
    assert np.array_equal(leafnode_sampler.sample_from_leaves_return_value,tree_method.missing_handler_.post_synth_transform_y)
    assert np.array_equal(result,missing_handling.post_synth_transform_result)

