import pytest
import pandas as pd
import numpy as np
from synthpop.synthesiser import Synthesiser
from synthpop.methods.cart_synth import CartMethod
import string

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

@pytest.mark.xfail(reason="known issue, see #130")
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


def simulate_realistic_dataset(n_samples=100):
    rng = np.random.default_rng(seed=852456)

    first_column = rng.random((n_samples,)) #first column is uniform random between 0 and 1.
    second_column = first_column*3 +5.5 + rng.random((n_samples,))*0.1 #Second column is linearly related to the first
    third_column = rng.choice(["a","b","c"],size=n_samples,replace=True)# third column is independent categorical

    #fourth column is correlated with both numeric and categoric variables.
    fourth_column = [first_column[i] if third_column[i] in ["a","b"] else second_column[i] for i in range(n_samples)] + rng.random((n_samples,))*0.1 

    #fifth column is categorial with many levels and correlated with both numeric and categorical columns
    # This is done by calculating a numeric value roughly between 0 and 26 and map that value to the alphabet.

    #The thrid column decides if the fifth is near the begin or the end of the alphabet
    distribution_general_means = [9 if third_column[i] in ["b","c"] else 18 for i in range(n_samples)]

    #The first column causes variance in the fifth column
    distribution_means = distribution_general_means + (first_column - 0.5)*6 

    alphabet_index = [int(rng.normal(distribution_means[i],6))%26 for i in range(n_samples)]

    fifth_column = [string.ascii_lowercase[alphabet_index[i]] for i in range(n_samples)]

    dataset = pd.DataFrame({
        "first":first_column,
        "second": second_column,
        "third":third_column,
        "fourth_column":fourth_column,
        "fifth":fifth_column
    })

#TODO: add missingness 

    return dataset
