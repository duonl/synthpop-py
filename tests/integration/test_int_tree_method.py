import pytest
import pandas as pd
import numpy as np

from synthpop.methods.cart_synth import TreeClassifierMethod

def test_treemethod_classifier_fit_and_transform():
    tree_method = TreeClassifierMethod()

    X = {
        "column1":np.array([1.1,2.2]),
        "column2":np.array([1.4,1.2]),
        "column3":np.array(["a","b"])
        }
    y = np.array(["x","y"])

    tree_method.fit(X,y)
    assert tree_method.n_features_in_ >= 3

    result = tree_method.transform(X)

    assert result.shape[0] ==2