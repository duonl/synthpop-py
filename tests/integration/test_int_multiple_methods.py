import pytest
import pandas as pd
import numpy as np

from synthpop.synthesiser import Synthesiser
from synthpop.methods.cart_synth import CartMethod
from synthpop.methods.copy_synth import CopyMethod
from synthpop.methods.sample_synth import SampleMethod

@pytest.mark.parametrize(
    "test_data",
    [

        (
            pd.DataFrame({
                "a": [1, 2]*10,
                "b": [3, 4]*10,
                "c": [5, 6]*10
            })
        ),

        (
            pd.DataFrame({
                "a": [np.nan, np.nan]*10,
                "b": [3, pd.NA]*10,
                "c": [None, 6]*10
            })
        ),

        (
            pd.DataFrame({
                "a": [1, None]*10,
                "b": [0, 0]*10,
                "c": [np.nan, np.nan]*10
            })  # Produces error for CART
        ),

        (
            pd.DataFrame({
                "a": [1, None]*10,
                "b": [0, -12]*10,
                "c": [pd.NA, pd.NA]*10
            })
        ),

        (
            pd.DataFrame({
                "a": [np.nan, None]*10,
                "b": [pd.NA, pd.NA]*10,
                "c": [0, -12]*10
            })
        ),

        (
            pd.DataFrame({
                "a": [pd.NA, pd.NA]*10,
                "b": [np.nan, np.nan]*10,
                "c": [0, -12]*10
            })
        )


    ]

)
def test_multiple_synthesis_methods(test_data):

    special_syn_method = {
        "a": SampleMethod(),
        "b": CopyMethod(),
        "c": CartMethod()
    }

    synth = Synthesiser(random_seed=2, special_syn_method=special_syn_method)
    fit = synth.fit(test_data)

    assert isinstance(fit.models_['a'], SampleMethod)
    assert isinstance(fit.models_['b'], CopyMethod)
    assert isinstance(fit.models_['c'], CartMethod)

    generated = fit.generate()

    assert test_data['b'].equals(generated['b'])
    expected_nan_values = test_data["c"].fillna(np.nan)
    # CART-method always outputs np.nan, but accepts pd.NA

    assert test_data['a'].isin(generated['a']).all()
    assert test_data['b'].isin(generated['b']).all()
    assert expected_nan_values.isin(generated['c']).all()

def test_copy_break():
    """Test if CopyMethod still produces an error if n != len(initial_dataset)"""
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6]
    })

    special_syn_method = {
        "a": SampleMethod(),
        "b": CopyMethod(),
        "c": CartMethod()
    }

    synth = Synthesiser(random_seed=2, special_syn_method=special_syn_method)
    fit = synth.fit(test_data)

    with pytest.raises(ValueError, match="Row mismatch"):
        fit.generate(n=10)
