import string

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from synthpop.synthesiser import Synthesiser
from synthpop.reproducibility import RandomStateManager

from sklearn.datasets import make_classification, make_regression

from synthpop.utils import str_dtype

def simulate_realistic_dataset_correlations(n_samples=100):
    rng = np.random.default_rng(seed=852456)

    # first column is uniform random between 0 and 1.
    first_column = rng.random((n_samples,))
    # Second column is linearly related to the first
    second_column = first_column*3 + 5.5 + rng.random((n_samples,))*0.1
    # third column is independent categorical
    third_column = rng.choice(["a", "b", "c"], size=n_samples, replace=True)

    # fourth column is correlated with both numeric and categoric variables.
    fourth_column = [first_column[i] if third_column[i] in [
        "a", "b"] else second_column[i] for i in range(n_samples)] + rng.random((n_samples,))*0.1

    # fifth column is categorial with many levels and correlated with both numeric and categorical columns
    # This is done by calculating a numeric value roughly between 0 and 26 and map that value to the alphabet.

    # The thrid column decides if the fifth is near the begin or the end of the alphabet
    distribution_general_means = [9 if third_column[i] in [
        "b", "c"] else 18 for i in range(n_samples)]

    # The first column causes variance in the fifth column
    distribution_means = distribution_general_means + (first_column - 0.5)*6

    alphabet_index = [int(rng.normal(distribution_means[i], 6)) %
                      26 for i in range(n_samples)]

    fifth_column = [string.ascii_lowercase[alphabet_index[i]]
                    for i in range(n_samples)]

    dataset = pd.DataFrame({
        "first": first_column,
        "second": second_column,
        "third": third_column,
        "fourth": fourth_column,
        "fifth": fifth_column
    })

    return (dataset, ["first", "second", "fourth"], ["third", "fifth"])

def make_data_missing(X):

    #We need a pattern of missingness that is different for each column
    # The missingness pattern should not be too predictable.

    for ik, k in enumerate(X.keys()):

        # The missingness is periodic. ever p-th element is missing.
        # The value of p decreases for each column.
        p = (len(X.keys())-ik) 

        values = [v if i % p !=1 else np.nan for i,v in enumerate(X[k])]
        if pd.api.types.is_numeric_dtype(X[k].dtype):
            X[k]=np.array(values)
        else:
            X[k] = np.array(values,dtype=str_dtype)

    return X
def get_test_data_regressor(seed = 10,with_cats=False,with_missing_features=False,with_missing_target=False):
    X,y = make_regression(random_state=seed)
    X = {i:X[:,i] for i in range(X.shape[1])}

    idx_cats = [3,4,6]
    if with_cats:
        for idx in idx_cats:
            x = (X[idx]*10).astype(int)
            x_i = [f %26 for f in x]
            X[idx] = np.array([string.ascii_lowercase[i] for i in x_i],dtype = str_dtype)

    if with_missing_features:
        X = make_data_missing(X)

    if with_missing_target:
        y = np.array([v if i%5 !=0 else np.nan for i,v in enumerate(y)])

    return (X,y)


#@pytest.mark.parametrize("seed",[(i) for i in range(50)])
def test_error_unseen_node():
    X,y = get_test_data_regressor(seed=0,with_cats=True,with_missing_features= True,with_missing_target=True)

    RandomStateManager.set_root_seed(0)
    obs = pd.DataFrame(X)
    obs["target"] = y

    synth = Synthesiser(random_seed=0)

    synth.fit(obs)

    synth.generate(100)


def test_reproducibilty_synthesis():

    RandomStateManager.set_root_seed([1])
    obs = simulate_realistic_dataset_correlations(n_samples=1000)[0][["first","second","third"]]

    synth = Synthesiser(random_seed=0)
    synth.fit(obs)

    syn1 = synth.generate(2000)
    syn2 = synth.generate(2000)

    assert syn1.equals(syn2)

    RandomStateManager.set_root_seed([1])
    synth2 = Synthesiser(random_seed=0)
    synth2.fit(obs)

    syn3 = synth2.generate(2000)

    for col in syn3.columns:
        assert (syn3[col] == syn2[col]).all(), f"column {col} not reproduced"