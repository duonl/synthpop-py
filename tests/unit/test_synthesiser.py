import pytest 
from synthpop.synthesiser import Synthesiser
from synthpop.methods.base_synth import BaseSynthMethod
import pandas as pd
import copy

class StubSynthMethod(BaseSynthMethod):
    pass
def test_synthesiser_default_synthesis(mocker):

    mock_cart = mocker.patch("synthpop.methods.cart_synth.CartMethod",spec=True)
    mock_cart.fit.return_value = copy.copy(mock_cart)

    
    mock_cart.__sklearn_clone__.return_value= copy.copy(mock_cart)
    
    test_data  = pd.DataFrame({
        "a":[1,2],
        "b":[3,4],
        "c":[5,6]
    })

    synth = Synthesiser(random_seed=2,default_syn_method=mock_cart)

    synth.fit(test_data)

    expected_initial_data = pd.DataFrame({"init":[0,0]})

    assert isinstance(synth.models_["a"],CartMethod)
    assert not synth.models_["a"] is mock_cart
    synth.models_["a"].fit.assert_any_call(expected_initial_data,test_data["a"])

    assert isinstance(synth.models_["b"],CartMethod)
    assert not synth.models_["b"] is mock_cart
    synth.models_["b"].fit.assert_called_with(test_data["a"],test_data["b"])

    assert isinstance(synth.models_["c"],CartMethod)
    assert not synth.models_["c"] is mock_cart
    synth.models_["c"].fit.assert_called_with(test_data[["a","b"]],test_data["b"])

    

    
