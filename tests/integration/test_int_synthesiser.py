import pytest
import pandas as pd
import numpy as np
from synthpop.synthesiser import Synthesiser
from synthpop.methods.cart_synth import CartMethod

def test_synthesiser_correct_default_methods():
    synth = Synthesiser(random_seed=2)
    test_data  = pd.DataFrame({
        "a":[1,2],
        "b":[3,4],
        "c":[5,6]
    })

    synth.fit(test_data)

    assert isinstance(synth.models_["a"],CartMethod)
    assert isinstance(synth.models_["b"],CartMethod)
    assert isinstance(synth.models_["c"],CartMethod)


def test_synthesiser_first_column_is_sampled_categorical():
    expected_proportions = {
        "a": 1/2,
        "b": 1/3,
        "missing": 1/6
    }

    n_samples = 300

    a_list = (["a"]*int(n_samples*expected_proportions["a"])) 
    b_list = (["b"]*int(n_samples*expected_proportions["b"])) 
    missing_list = ([np.nan]*int(n_samples*expected_proportions["missing"])) 
    test_data = pd.DataFrame({
        "first_column": a_list + b_list+missing_list
    })

    synth = Synthesiser(random_seed=2)

    result = synth.fit(test_data).generate()

    result_proportions = result["first_column"].value_counts(dropna = False,normalize=True)

    assert np.abs(expected_proportions["a"] - result_proportions["a"])<0.05
    assert np.abs(expected_proportions["b"] - result_proportions["b"])<0.05
    assert np.abs(expected_proportions["missing"] - result_proportions[np.nan])<0.05

def test_synthesiser_first_column_is_sampled_numeric():
    expected_proportions = {
        '1.1': 1/2,
        '2': 1/3,
        "missing": 1/6
    }

    n_samples = 3000

    one_list = ([1.1]*int(n_samples*expected_proportions["1.1"])) 
    two_list = ([2]*int(n_samples*expected_proportions["2"])) 
    missing_list = ([np.nan]*int(n_samples*expected_proportions["missing"])) 
    test_data = pd.DataFrame({
        "first_column": one_list + two_list+missing_list
    })

    synth = Synthesiser(random_seed=2)

    result = synth.fit(test_data).generate()

    result_proportions = result["first_column"].value_counts(dropna = False,normalize=True)

    assert np.abs(expected_proportions["1.1"] - result_proportions[1.1])<0.05
    assert np.abs(expected_proportions["2"] - result_proportions[2])<0.05
    assert np.abs(expected_proportions["missing"] - result_proportions[np.nan])<0.05