import pandas as pd
import numpy as np
import numpy.typing as npt
import pytest
from sklearn import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from synthpop.data_processing.missing_value_handling import BaseMissingValueHandler
from synthpop.methods.cart_synth import _AbstractTreeMethod,TreeClassifierMethod, TreeRegressorMethod, _to_fixed_length_string_array
from sklearn.utils.estimator_checks import parametrize_with_checks

import copy
import pytest

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from synthpop.data_processing.missing_value_handling import BaseMissingValueHandler
from synthpop.methods.cart_synth import _AbstractTreeMethod,TreeClassifierMethod, TreeRegressorMethod, _to_fixed_length_string_array

str_dtype = np.dtypes.StringDType(na_object=np.nan)

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
    

@pytest.fixture
def encoder():
    #The result of the transform of encoding is always a 2D np.array of float32, with one or more columns
    return TransformStub(transform_return_value=np.array([[1.1],[2.2],[3.3],[4.4],[5.5],[6.6]]))


class StubMissingHandler(BaseMissingValueHandler):
    
    def __init__(self, prepared_for_fit_result,post_synth_transform_result):
        self.prepared_for_fit_result = prepared_for_fit_result
        self.post_synth_transform_result = post_synth_transform_result
        self.clone_called = False
    
    def prepare_data_for_fit(self, X, y):
        self.prepare_data_for_fit_X = X
        self.prepare_data_for_fit_y = y
        return self.prepared_for_fit_result
    
    def post_synth_transform(self, X, y):
        self.post_synth_transform_X = X
        self.post_synth_transform_y = y
        return self.post_synth_transform_result
    
    def clone(self):
        self.clone_called = True
        return copy.copy(self)
    
    def __sklearn_clone__(self):
        return copy.copy(self)

@pytest.fixture
def missing_handling(request):

    # The result X of prepare_data_for_fit must contain the same columns as X
    X = request.node.callspec.params["X"]
    X_prepare_res = {k:np.array([k]*3) for k in X.keys()}
    y_prepare_res = np.arange(0,3)

    y_post_synthesis_result = np.arange(3,6)
    return StubMissingHandler(prepared_for_fit_result=(X_prepare_res,y_prepare_res),post_synth_transform_result=y_post_synthesis_result)



class StubLeafNodeSampler():
    def __init__(self,sample_from_leaves_return_value):
        self.sample_from_leaves_return_value = sample_from_leaves_return_value
        self.clone_called = False
    
    def fit_sampler(self, leaf_ids: npt.ArrayLike, y: npt.ArrayLike):
        self.fit_sampler_leaf_ids = leaf_ids
        self.fit_sampler_y = y
        return self
    
    def sample_from_leaves(self, leaf_ids: npt.ArrayLike) -> np.ndarray:
        self.sample_from_leaves_leaf_ids = leaf_ids
        return self.sample_from_leaves_return_value
    
    def clone(self):
        self.clone_called = True
        return copy.copy(self)

@pytest.fixture
def leafnode_sampler():
    return StubLeafNodeSampler(sample_from_leaves_return_value=np.array([1,2,3,4]))


class StubTree():
    def __init__(self,apply_result=None):
        self.apply_result=apply_result
        pass

    def fit(self,X,y):
        self.fit_X_ = X
        self.fit_y_ = y
        return self
    
    def apply(self,X):
        self.apply_X_ = X
        return self.apply_result
    def __sklearn_clone__(self):
        return copy.copy(self)

class TestTreeMethod(_AbstractTreeMethod):
    def __init__(self, *,
                 encoder = None,
                missing_handling = None,
                tree_sampler = None,
                tree = None):
        super().__init__(encoder=encoder, missing_handler=missing_handling, tree_sampler=tree_sampler,tree=tree)


    def _get_encoder(self):
        return TransformStub()
    
    def _get_tree(self):
        return StubTree()
    
    def _get_missing_handling(self):
        return super()._get_missing_handling()
    
@pytest.fixture
def tree_method(encoder,missing_handling,leafnode_sampler):
    return TestTreeMethod(encoder=encoder,missing_handling=missing_handling,tree_sampler=leafnode_sampler,tree=StubTree())

@pytest.fixture()
def fitted_tree(tree_method,request):

    X = request.node.callspec.params["X"]
    cat_index = request.node.callspec.params["index_cat"]
    tree_method.encoders_ =  {k:clone(tree_method.encoder) for k in cat_index}
    tree_method.missing_handler_ = tree_method.missing_handler.clone()
    tree_method.tree_sampler_ = tree_method.tree_sampler.clone()
    tree_method.tree_ = clone(tree_method.tree)
    tree_method.n_features_in_ = len(X.keys())

    tree_method.feature_order_ = list(X.keys())

    return tree_method

@pytest.fixture(autouse=True)
def mock_fit_decision_tree(mocker,):
    mocker.patch("synthpop.methods.tree_utils._fit_decision_tree_consistently",return_value = StubTree())

#---------------------------------------------------

def assert_dict_array_equal(expected,actual):

    for (k,v) in expected.items():
        assert np.array_equal(v,actual[k],equal_nan=True), f"expected (key = {k}): {v}. Actual: {actual[k]}"

    assert len(expected.keys()) == len(actual.keys()), f"actual has more keys than expected. Expected: {len(expected.keys())}. Actual: {actual.keys()}"

# Test data ---------------------------------------------------------------------------------------


def get_input_test_data():

    X1 = {
        "num_1":np.array([1,2,np.nan,2,5,3]),
        "cat_1":np.array(["a","b","c","d","e","f"],dtype=str_dtype),
        "num_2":np.array([1.1,2.2,5.5,2.2,5.5,3.3]),
        "cat_2":np.array(["aa","bc","cc","dD","eE","fF"],dtype=str_dtype),
    }

    X2 = {
        "num_1":np.array([1,2,5,2,5,3]),
        "num_2":np.array([1.1,2.2,5.5,2.2,5.5,3.3]),
    }

    X3 = {
        "cat_1":np.array(["a","b","c","d","e","f"]),
        "cat_2":np.array(["aa","bc","cc","dD","eE","fF"]),
    }

    y1 = np.array([1.2,2.3,3.4,5.6,7.8,8.9])
    y2 = np.array(["a","b","c","d","e","f"],dtype=str_dtype)


    # Some tests need a ground truth of which columns are categorical.
    # That is why a list of categorical columns is supplied to each test.
    base_cases_numpy = [ (X1,y1,["cat_1","cat_2"]),
                        (X2,y2,[]),
                        (X3,y1,["cat_1","cat_2"]),
                        ]

    return base_cases_numpy

def get_exp_feature_matrix():
    return np.array([[1,2],[3,4]])
# Fixtures ----------------------------------------------------------------------------------------


@pytest.fixture
def apply_result(missing_handling):
    n = len(missing_handling.prepared_for_fit_result[0])
    apply_result = np.array([i%3 for i in range(n)])
    return apply_result


@pytest.fixture(autouse=True)
def stub_build_feature_matrix(request,monkeypatch):
    # The test should not use the real implementation of build_feature_matrix.
    # The test should use a stub of that function instead.
    # There is one exception to this: the standard tests of sklearn do need the real build_feature_matrix.

    # setting autouse=True causes this fixture to be used in every test.
    # to enable the exception, we check for the 'noautofixt' mark.
    if 'noautofixt' in request.keywords:
        return
    
    #We need to import tree utils here so that we can replace build_feature_matrix with our stub.
    from synthpop.methods import tree_utils

    # We define the stub
    def stub_build_feature_matrix(X,feature_order):
        return get_exp_feature_matrix()
    
    # and use monkey patching to replace the method with the stub.
    monkeypatch.setattr(tree_utils,"build_feature_matrix",stub_build_feature_matrix)

@pytest.fixture(autouse=True)
def stub_validate_dict(request,monkeypatch):

    # validating the input is delegated to functions in utils.py.
    # So for these unit test, we need to stub those functions

    # setting autouse=True causes this fixture to be used in every test.
    # to enable the exception, we check for the 'noautofixt' mark.
    if 'noautofixt' in request.keywords:
        return
    
    #We need to import tree utils here so that we can replace build_feature_matrix with our stub.
    from synthpop import utils

    # and use monkey patching to replace the method with the stub.
    monkeypatch.setattr(utils,"validate_2d_dict",lambda X: (X,42))
    monkeypatch.setattr(utils,"validate_1d_target",lambda y,n_samples: y)


# test fit ----------------------------------------------------------------------------------------
# Note: raising errors on bad shapes is delegated to utils.validate_dict_x and utils.validate_y
# In the unit tests of tree methods, we only check that de delegation happens.
@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_validates_X_and_y(X,y,index_cat,tree_method,mocker):
    from synthpop import utils

    X_spy = mocker.spy(utils,"validate_2d_dict")
    y_spy = mocker.spy(utils,"validate_1d_target")

    tree_method.fit(X,y)

    X_spy.assert_called_once_with(X)
    y_spy.assert_called_once_with(y,42)#42 is the hardcoded n_samples value of the validate_dict_x stub

@pytest.mark.parametrize("X, y, index_cat", get_input_test_data())
def test_fit_clones_dependencies_correctly(X, y, index_cat, tree_method):

    tree_method.fit(X,y)
    assert tree_method.missing_handler.clone_called
    assert tree_method.tree_sampler.clone_called




@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_n_features_in(X,y,index_cat,tree_method):

    tree_method.fit(X, y)
    assert tree_method.n_features_in_ == len(X)

@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_trains_encoder(X,y,index_cat,tree_method):

    
    tree_method.fit(X,y)

    for i in index_cat:
        assert np.array_equal(tree_method.encoders_[i].fit_X_,X[i])
        assert np.array_equal(tree_method.encoders_[i].fit_y_,y)
        assert not ( tree_method.encoders_[i] is tree_method.encoder)

        for i2 in index_cat:
            assert (not ( tree_method.encoders_[i] is tree_method.encoders_[i2])) or i2==i

@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_transforms_with_encoder(X,y,index_cat,tree_method):


    tree_method.fit(X,y)

    for i in index_cat:
        assert np.array_equal(tree_method.encoders_[i].transform_X_,tree_method.missing_handler_.prepared_for_fit_result[0][i])


@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_prepare_data_for_fit_is_called(X,y,index_cat,tree_method):

    tree_method.fit(X,y)

    assert_dict_array_equal(expected=X,actual=tree_method.missing_handler_.prepare_data_for_fit_X)

    assert np.array_equal(tree_method.missing_handler_.prepare_data_for_fit_y,y)
    assert not (tree_method.missing_handler_ is tree_method.missing_handler)

@pytest.mark.parametrize("X,y",[(v[0],v[1]) for v in get_input_test_data()])
def test_fit_sets_order(X,y,tree_method):

    tree_method.fit(X,y)

    assert list(X.keys()) == tree_method.feature_order_

@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_build_feature_matrix(X,y,index_cat,tree_method,mocker):
    from synthpop.methods import tree_utils

    spy = mocker.spy(tree_utils,"build_feature_matrix")

    tree_method.fit(X,y)
    X_exp = {k: tree_method.encoders_[k].transform_return_value if k in index_cat else v for (k,v) in tree_method.missing_handler_.prepared_for_fit_result[0].items()}

    expected_order = tree_method.feature_order_
    spy.assert_called_once_with(X_exp,expected_order)
    

@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_tree_is_fit(X,y,index_cat,tree_method,mocker):

    expected_tree = clone(tree_method.tree)

    mock_fit_decision_tree_consistently = mocker.patch("synthpop.methods.tree_utils._fit_decision_tree_consistently",return_value = expected_tree)


    tree_method.fit(X,y)

    expected_X = get_exp_feature_matrix()
    expected_y = tree_method.missing_handler_.prepared_for_fit_result[1]

    assert len(mock_fit_decision_tree_consistently.mock_calls) == 1, "fit_decision_tree_consistently should be called 1 time"

    kwargs = mock_fit_decision_tree_consistently.mock_calls[0][2]

    assert isinstance(kwargs["decision_tree"], StubTree)
    assert tree_method.tree_ is expected_tree, "fitted decision tree should be stored"

    assert np.array_equal(kwargs["X"],expected_X,equal_nan=True)
    assert np.array_equal(kwargs["y"],expected_y,equal_nan=True)

    



@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_tree_is_applied(X,y,index_cat,tree_method):

    tree_method.fit(X,y)

    assert np.array_equal(get_exp_feature_matrix(),tree_method.tree_.apply_X_,equal_nan=True)


@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_sampler_fit(X,y,index_cat,tree_method):
    tree_method.fit(X,y)

    assert np.array_equal(tree_method.tree_sampler_.fit_sampler_leaf_ids,tree_method.tree_.apply_result), "input of the sampler must be the output of the tree"
    assert np.array_equal(tree_method.tree_sampler_.fit_sampler_y,tree_method.missing_handler_.prepared_for_fit_result[1])
    assert not (tree_method.tree_sampler is tree_method.tree_sampler_)

@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_set_feature_names_out(X,y,index_cat,tree_method):

    y = pd.Series(y,name="target_name")

    tree_method.fit(X,y)

    assert tree_method.target_name_ == "target_name"

@pytest.mark.parametrize("X,y,index_cat",get_input_test_data())
def test_fit_set_feature_names_out_no_target_name(X,y,index_cat,tree_method):


    tree_method.fit(X,y)

    assert tree_method.target_name_ is None

def test_fit_classifier_converts_to_str(encoder,leafnode_sampler,mocker):
    X = {"a":np.array([1,2])}
    y = np.array(["a","b"],dtype=str_dtype)

    mock_fit_decision_tree_consistently = mocker.patch("synthpop.methods.tree_utils._fit_decision_tree_consistently",return_value =StubTree())
    # the missing handling can return a y of str_dtype.
    missing_handling = StubMissingHandler(prepared_for_fit_result=(X,y),post_synth_transform_result=None)
    
    str_y = np.array(["x","y"])
    mocked_to_str= mocker.patch('synthpop.methods.cart_synth._to_fixed_length_string_array',return_value=str_y)

    tree_method = TreeClassifierMethod(encoder=encoder,missing_handler=missing_handling,tree_sampler=leafnode_sampler,tree=StubTree())
    

    tree_method.fit(X,y)

    mocked_to_str.assert_called_with(y)

    actual_y = mock_fit_decision_tree_consistently.mock_calls[0][2]["y"]
    assert np.array_equal(str_y,actual_y)
    
def test_fit_regressor_converts_to_float32(encoder,leafnode_sampler,mocker):
    X = {"a":np.array([1,2])}
    y = np.array([1,2.0],dtype=np.float64)

    mock_fit_decision_tree_consistently = mocker.patch("synthpop.methods.tree_utils._fit_decision_tree_consistently",return_value =StubTree())

    # the missing handling can return a y of np.float64
    missing_handling = StubMissingHandler(prepared_for_fit_result=(X,y),post_synth_transform_result=None)
    
    converted_y = np.array([1,2.0],dtype=np.float32)

    tree_method = TreeRegressorMethod(encoder=encoder,missing_handler=missing_handling,tree_sampler=leafnode_sampler,tree=StubTree())
    

    tree_method.fit(X,y)

    actual_y = mock_fit_decision_tree_consistently.mock_calls[0][2]["y"]

    assert np.array_equal(converted_y,actual_y)
    assert actual_y.dtype == np.float32

    

# test transform ----------------------------------------------------------------------------------



@pytest.mark.parametrize("X,index_cat",[(v[0],v[2]) for v in get_input_test_data()])
def test_transform_encodes_data(X,index_cat,fitted_tree):

    fitted_tree.transform(X)

    for i in index_cat:
        assert isinstance(fitted_tree.encoders_[i].transform_X_,np.ndarray)
        assert np.array_equal(X[i],fitted_tree.encoders_[i].transform_X_)


@pytest.mark.parametrize("X,index_cat",[(v[0],v[2]) for v in get_input_test_data()])
def test_transform_build_feature_matrix(X,index_cat,fitted_tree,mocker):
    from synthpop.methods import tree_utils

    spy = mocker.spy(tree_utils,"build_feature_matrix")
    tree_method = fitted_tree 

    # we need to verify that the feature order seen when fitting is used.
    # setting feature_order_ to a random sequence guarantees that the tree method uses feature_order_ and does not use list(X.keys())
    tree_method.feature_order_ = ["some","random","order"]

    tree_method.transform(X)
    X_exp = {k: tree_method.encoders_[k].transform_return_value if k in index_cat else v for (k,v) in X.items()}
    spy.assert_called_once_with(X_exp,tree_method.feature_order_ )

@pytest.mark.parametrize("X,index_cat",[(v[0],v[2]) for v in get_input_test_data()])
def test_transform_applies_fitted_tree(X,index_cat,fitted_tree):
    tree_method =fitted_tree

    tree_method.transform(X)

    expected_input_for_tree = get_exp_feature_matrix()

    assert np.array_equal(expected_input_for_tree,tree_method.tree_.apply_X_,equal_nan=True)



@pytest.mark.parametrize("X,index_cat",[(v[0],v[2]) for v in get_input_test_data()])
def test_transform_uses_sampler(X,index_cat,fitted_tree):

    tree_method =fitted_tree

    tree_method.transform(X)

    assert np.array_equal(tree_method.tree_.apply_result,tree_method.tree_sampler_.sample_from_leaves_leaf_ids)

@pytest.mark.parametrize("X,index_cat",[(v[0],v[2]) for v in get_input_test_data()])
def test_transform_calls_post_synth_transform(X,index_cat,fitted_tree):
    
    tree_method = fitted_tree

    result = tree_method.transform(X)

    assert_dict_array_equal(X, tree_method.missing_handler_.post_synth_transform_X)
    assert np.array_equal(tree_method.tree_sampler_.sample_from_leaves_return_value,tree_method.missing_handler_.post_synth_transform_y)
    assert np.array_equal(result, tree_method.missing_handler_.post_synth_transform_result)

@pytest.mark.parametrize("X",[v[0] for v in get_input_test_data()])
def test_transform_raises_error_when_not_fitted(X,tree_method):
    
    with pytest.raises(NotFittedError):
        tree_method.transform(X)

def test_regressor_transform_returns_float32(leafnode_sampler):
    X = {"a":np.array([1,2])}
    y = np.array([1,2.0],dtype=np.float64)

    # the missing handling can return a y of np.float64
    missing_handling = StubMissingHandler(prepared_for_fit_result=None,post_synth_transform_result=y)
    
    tree_method = TreeRegressorMethod(encoder=None,missing_handler=missing_handling,tree_sampler=leafnode_sampler,tree=StubTree())
    tree_method.encoders_ =  {}
    tree_method.missing_handler_ = missing_handling
    tree_method.tree_sampler_ = leafnode_sampler
    tree_method.tree_ = StubTree()
    tree_method.n_features_in_ = len(X.keys())
    tree_method.feature_order_ = list(X.keys())
    

    result = tree_method.transform(X)

    assert np.array_equal(result,tree_method.missing_handler_.post_synth_transform_result)
    assert result.dtype == np.float32


def test_classifier_transform_returns_str_dtype(leafnode_sampler):
    """
    The output of the classifier method should ALWAYS be str_dtype.
    With the default values of the TreeClassifierMethod, no explicit conversion is needed.
    However, if MissingValuePredictor is used as missing handler, it might happen that the output is not already str_dtype.
    Since the consequence of such an unexpected dtype could be that np.nan becomes "NaN" (which would be a silent error), we need to explicitly deal with this situation.
    """
    
    X = {"a":np.array([1,2])}
    y = np.array(["a","b"],dtype=np.str_)

    # the missing handling can return a y of np.float64
    missing_handling = StubMissingHandler(prepared_for_fit_result=None,post_synth_transform_result=y)
    
    tree_method = TreeClassifierMethod(encoder=None,missing_handler=missing_handling,tree_sampler=leafnode_sampler,tree=StubTree())
    tree_method.encoders_ =  {}
    tree_method.missing_handler_ = missing_handling
    tree_method.tree_sampler_ = leafnode_sampler
    tree_method.tree_ = StubTree()
    tree_method.n_features_in_ = len(X.keys())
    tree_method.feature_order_ = list(X.keys())
    

    result = tree_method.transform(X)

    assert np.array_equal(result,tree_method.missing_handler_.post_synth_transform_result)
    assert result.dtype == str_dtype


#general tests ------------------------------------------------------------------------------------

@pytest.mark.parametrize("X",[v[0] for v in get_input_test_data()])
def test_get_feature_names_out(X,tree_method):
    tree_method.target_name_ = "name_of_target"

    result = tree_method.get_feature_names_out()
    assert result == ["name_of_target"]

@pytest.mark.parametrize("X",[v[0] for v in get_input_test_data()])
def test_get_feature_names_out_no_target_name(X,tree_method):
    tree_method.target_name_ = None
    tree_method.feature_order_ = ["Trained","on","these","features"]

    result = tree_method.get_feature_names_out()
    assert result == [["Trained","on","these","features"]]


def ndarray_to_dict(a):
    if isinstance(a,np.ndarray):
        return {i: a[:,i] for i in range(a.shape[1])}
    return a
@parametrize_with_checks([TreeClassifierMethod(),TreeRegressorMethod()], legacy=False, expected_failed_checks=lambda x: {
    "check_fit_score_takes_y":"tests with a score component"
})
@pytest.mark.noautofixt
def test_TreeMethod_is_sklearn_compatible(estimator, check):
    # sklearn provides valuable tests.
    # Those test assume that the input is a numpy array.
    # The tree methods assume that the input is a dictionary.

    # We want to test if the tree method that the user is going to use are sklearn compatible.
    # So we cannot use the TestTreeMethod as in all other tests.

    # The solution is that a class is constructed in each sklearn test.
    # This class inherits from the applicable tree method, and overwrites the fit and transform to convert np arrays to dictionaries.
    class EstimatorWrap(estimator.__class__):
        def fit(self,X,y):
            return super().fit(ndarray_to_dict(X),y)
        
        def transform(self,X):
            return super().transform(ndarray_to_dict(X))
        
    #This is needed to change the datatype of the estimator to the child class.
    #estimator.__class__ = EstimatorWrap
    wrapped = EstimatorWrap(**estimator.get_params())

    check(wrapped)

def test_to_fixed_lenght_string_array():
    x = np.array(["a","b"],dtype=str_dtype)
    result = _to_fixed_length_string_array(x)

    assert result.dtype == "U1"
    assert np.array_equal(result,["a","b"])

    x = np.array(["aa","bb","c"],dtype=str_dtype)
    result = _to_fixed_length_string_array(x)

    assert result.dtype == "U2"
    assert np.array_equal(result,["aa","bb","c"])
