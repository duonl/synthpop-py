import numpy as np
import pandas as pd
import pytest

from synthpop.methods.cart_synth import (
    CartMethod,
    TreeClassifierMethod,
)
from synthpop.data_processing.missing_value_handling import MissingValuePredictor

@pytest.mark.parametrize(
    "y, none_missing",
    [
        (pd.Series(['a', 'b', np.nan, 'c'] * 10, dtype=str, name="target"), False),
        (pd.Series(['a', 'b', 'N.a.N', 'c'] * 10, dtype=str, name="target"), True),
        (pd.Series(['a', 'b', np.nan, 'c'] * 10,
         dtype=object, name="target"), False),
    ]
)
def test_classifier_method_and_replace_missing_with_value(y, none_missing):
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50] * 10,
            "income": [1000.0, 2000.0, 3000.0, 4000.0] * 10,
            "blood type": ["A", "O", "AB", "O"] * 10,
        }
    )

    cart = CartMethod(
        classifier=TreeClassifierMethod(
            tree=None, missing_handler=MissingValuePredictor()
        ),
    )

    cart.fit(X, y)

    assert isinstance(cart.method_.missing_handler_, MissingValuePredictor)
    assert not cart.method_.missing_handler_._all_missing
    assert cart.method_.missing_handler_._none_missing == none_missing

    out = cart.transform(X)

    assert isinstance(out, pd.Series)
    assert len(out) == len(X)
    assert out.name == "target"