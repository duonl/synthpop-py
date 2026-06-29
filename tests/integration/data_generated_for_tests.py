import copy
import pytest
import pandas as pd
import numpy as np
import string

from synthpop.utils import str_dtype
from sklearn.datasets import make_classification, make_regression


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


def get_test_data_classifier(seed = 10,n_samples=100,with_cats = False,with_missing_features=False,with_missing_target=False):
    X,y = make_classification(random_state=seed,n_samples=n_samples,n_classes=10,n_informative=11)
    
    X = {i:X[:,i] for i in range(X.shape[1])}


    idx_cats = [3,4,6]
    if with_cats:
        for idx in idx_cats:
            x = (X[idx]*10).astype(int)
            x_i = [f %5 for f in x]
            X[idx] = np.array([string.ascii_lowercase[i%26] for i in x_i],dtype=str_dtype)

    if with_missing_features:
        X = make_data_missing(X)

    if with_missing_target:
        y = np.array([string.ascii_lowercase[i%26] if i%5 !=0 else np.nan for i in y],dtype=str_dtype)
    else:
        y = np.array([string.ascii_lowercase[i%26] for i in y],dtype=str_dtype)
    return (X,y)


def get_test_data_regressor(seed = 10,n_samples=100,with_cats=False,with_missing_features=False,with_missing_target=False):
    X,y = make_regression(random_state=seed,n_samples=n_samples)
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