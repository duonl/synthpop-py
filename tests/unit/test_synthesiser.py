import pytest 
from synthpop.synthesiser import Synthesiser
from synthpop.methods.base_synth import BaseSynthMethod
import pandas as pd
import copy

class StubSynthMethod(BaseSynthMethod):

    def __init__(self,transform_result=None):
        super().__init__()
        self.transform_result = transform_result
        self.fit_X = []
        self.fit_y = []
        self.transform_X = []

    def fit(self, X, y):

        self.fit_X = self.fit_X + [X]
        self.fit_y = self.fit_y + [y]

        return self
    
    def transform(self, X):
        self.transform_X = self.transform_X + [copy.copy(X)]
        return self.transform_result
    
    def get_feature_names_out(self, input_features=None):
        raise Exception()
        return ""

def assert_fit_call(model,expected_X,expected_y,expected_model):
    assert isinstance(model,expected_model)
    assert isinstance(model.fit_X[0],pd.DataFrame)
    assert model.fit_X[0].equals(expected_X)
    assert model.fit_y[0].equals(expected_y)
    assert len(model.fit_X) == 1, "fitting should happen 1 time per column"

def assert_distinct_instances(objects,origin):
    for a in objects:
        assert not (objects[a] is origin), "instance should not be original"
        for b in objects:

            if a == b:
                continue
            
            assert not (objects[a] is objects[b]), "instances are not distinct"
def test_synthesiser_fit_default_synthesis():

    synth_method = StubSynthMethod()
    test_data  = pd.DataFrame({
        "a":[1,2],
        "b":[3,4],
        "c":[5,6]
    })

    synth = Synthesiser(random_seed=2,default_syn_method=synth_method)

    synth.fit(test_data)

    assert synth.n_samples_ == 2
    assert synth.column_order_ == ["a","b","c"]

    expected_initial_data = pd.DataFrame({"init":[0,0]})

    assert_fit_call(synth.models_["a"],expected_X=expected_initial_data,expected_y=test_data["a"],expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["b"],expected_X=test_data[["a"]],expected_y=test_data["b"],expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["c"],expected_X=test_data[["a","b"]],expected_y=test_data["c"],expected_model=StubSynthMethod)

    assert_distinct_instances(synth.models_,origin=synth_method)


def test_synthesiser_fit_custom_order_by_column_name():

    synth_method = StubSynthMethod()
    test_data  = pd.DataFrame({
        "a":[1,2],
        "b":[3,4],
        "c":[5,6]
    })

    synth = Synthesiser(random_seed=2,default_syn_method=synth_method,column_order=["b","a","c"])

    synth.fit(test_data)

    assert synth.n_samples_ == 2
    assert synth.column_order_ == ["b","a","c"]

    expected_initial_data = pd.DataFrame({"init":[0,0]})

    assert_fit_call(synth.models_["b"],expected_X=expected_initial_data,expected_y=test_data["b"],expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["a"],expected_X=test_data[["b"]],expected_y=test_data["a"],expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["c"],expected_X=test_data[["b","a"]],expected_y=test_data["c"],expected_model=StubSynthMethod)

    assert_distinct_instances(synth.models_,origin=synth_method)

def test_synthesiser_fit_custom_order_by_column_index():

    synth_method = StubSynthMethod()
    test_data  = pd.DataFrame({
        "a":[1,2],
        "b":[3,4],
        "c":[5,6]
    })

    synth = Synthesiser(random_seed=2,default_syn_method=synth_method,column_order=[2,1,0])

    synth.fit(test_data)

    assert synth.n_samples_ == 2
    assert synth.column_order_ == ["c","b","a"]

    expected_initial_data = pd.DataFrame({"init":[0,0]})

    assert_fit_call(synth.models_["c"],expected_X=expected_initial_data,expected_y=test_data["c"],expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["b"],expected_X=test_data[["c"]],expected_y=test_data["b"],expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["a"],expected_X=test_data[["c","b"]],expected_y=test_data["a"],expected_model=StubSynthMethod)

    assert_distinct_instances(synth.models_,origin=synth_method)



def test_generate_default():

    synth = Synthesiser(random_seed=2)

    expected_result = pd.DataFrame({
        "a":["x","y","z"],
        "b":[1,2,3],
        "c":["q","w","e"]
    })

    expected_initial_data = pd.DataFrame({"init":[0,0,0]})

    synth.column_order_ = ["a","b","c"]
    synth.n_samples_ = 3
    synth.models_ = {}
    synth.models_["a"] = StubSynthMethod(transform_result=expected_result["a"])
    synth.models_["b"] = StubSynthMethod(transform_result=expected_result["b"])
    synth.models_["c"] = StubSynthMethod(transform_result=expected_result["c"])

    result = synth.generate()
    assert isinstance(result,pd.DataFrame)
    assert expected_result.equals(result)

    assert synth.models_["a"].transform_X[0].equals(expected_initial_data)
    assert synth.models_["b"].transform_X[0].equals(expected_result[["a"]])
    assert synth.models_["c"].transform_X[0].equals(expected_result[["a","b"]])
    

    
