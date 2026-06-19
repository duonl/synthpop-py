import string

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from synthpop.synthesiser import Synthesiser
from synthpop.reproducibility import RandomStateManager

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


def test_reproducibilty_synthesis():

    RandomStateManager.set_root_seed(1)
    obs = simulate_realistic_dataset_correlations(n_samples=1000)[0][["first","second","third"]]

    synth = Synthesiser(random_seed=0)
    synth.fit(obs)

    syn1 = synth.generate(2000)
    syn2 = synth.generate(2000)

    assert syn1.equals(syn2)

    RandomStateManager.set_root_seed(1)
    synth2 = Synthesiser(random_seed=0)
    synth2.fit(obs)

    syn3 = synth2.generate(2000)

    for col in syn3.columns:
        assert (syn3[col] == syn2[col]).all(), f"column {col} not reproduced"