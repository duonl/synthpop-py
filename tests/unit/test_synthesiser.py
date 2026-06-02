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
        self.transform_X = self.transform_X + [X]
        return self.transform_result
    
    def get_feature_names_out(self, input_features=None):
        raise Exception()
        return ""

def test_synthesiser_default_synthesis(mocker):

    
    synth_method = StubSynthMethod()
    test_data  = pd.DataFrame({
        "a":[1,2],
        "b":[3,4],
        "c":[5,6]
    })

    synth = Synthesiser(random_seed=2,default_syn_method=synth_method)

    synth.fit(test_data)

    expected_initial_data = pd.DataFrame({"init":[0,0]})

    assert isinstance(synth.models_["a"],StubSynthMethod)
    assert not synth.models_["a"] is synth_method
    assert isinstance(synth.models_["a"].fit_X[0],pd.DataFrame)
    assert synth.models_["a"].fit_X[0].equals(expected_initial_data)
    assert synth.models_["a"].fit_y[0].equals(test_data["a"])
    assert len(synth.models_["a"].fit_X) == 1, "fitting should happen 1 time per column"


    assert isinstance(synth.models_["b"],StubSynthMethod)
    assert not synth.models_["b"] is synth_method
    assert not synth.models_["b"] is synth.models_["a"]
    assert isinstance(synth.models_["b"].fit_X[0],pd.DataFrame)
    assert synth.models_["b"].fit_X[0].equals(test_data[["a"]]), "the first column should be a predictor for the second column"
    assert synth.models_["b"].fit_y[0].equals(test_data["b"])
    assert len(synth.models_["b"].fit_X) == 1, "fitting should happen 1 time per column"

    assert isinstance(synth.models_["c"],StubSynthMethod)
    assert not synth.models_["c"] is synth_method
    assert not synth.models_["c"] is synth.models_["a"]
    assert not synth.models_["c"] is synth.models_["b"]
    assert isinstance(synth.models_["c"].fit_X[0],pd.DataFrame)
    assert synth.models_["c"].fit_X[0].equals(test_data[["a","b"]]), "the first column should be a predictor for the second column"
    assert synth.models_["c"].fit_y[0].equals(test_data["c"])
    assert len(synth.models_["c"].fit_X) == 1, "fitting should happen 1 time per column"

    

    
