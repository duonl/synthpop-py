import pytest
import pandas as pd
import numpy as np

from synthpop.synthesiser import Synthesiser
from synthpop.methods.cart_synth import CartMethod
from synthpop.methods.copy_synth import CopyMethod
from synthpop.methods.sample_synth import SampleMethod

"""
Ideeen voor mezelf: CopyMethod foutmelding generate(n= niet input), 
geef ook aan dat t in CopyMethod is

CopyMethod accepteert wel pd.NA en output dat. CART niet. Inconsistent?

""" 

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
    test_data['c'][test_data['c'].isna()] = np.nan 
    # CART-method always outputs np.nan, but accepts pd.NA

    for col in test_data.columns:
        assert test_data[col].isin(generated[col]).all()


def test_copy_break():
    """Test if CopyMethod still produces an error if n != len(initial)dataset)"""
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


@pytest.mark.parametrize(
    "test_data",
    [
        (
            pd.DataFrame({
                "a": [1, np.nan, 4, np.nan, 2, 3, np.nan]*10,
                "b": ['x', 'y', 'x', 'y', 'x', 'x', 'y']*10,
            })
        ),

        (
            pd.DataFrame({
                "a": [1, pd.NA, 4, pd.NA, 2, 3, pd.NA]*10,
                "b": ['x', 'y', 'x', 'y', 'x', 'x', 'y']*10,
            })
        ),

        (
            pd.DataFrame({
                "a": [1, None, 4, None, 2, 3, None]*10,
                "b": ['x', 'y', 'x', 'y', 'x', 'x', 'y']*10,
            })
        ),
    ]
)
def test_CART_entire_nan_predictions(test_data):
    """Test the case where nan always implies-->y"""
    synth = Synthesiser(random_seed=2)
    fit = synth.fit(test_data).generate(n=100)

    valid = (
        ((fit["b"] == "y") & (fit["a"].isna())) |
        ((fit["b"] == "x") & (~fit["a"].isna()))
    )

    assert valid.all()


@pytest.mark.parametrize(
    "test_data",
    [
        (
            pd.DataFrame({
                "a": ['x', 'x', 'x', 'y', 'x', 'x', 'y']*10,
                "b": [np.nan, 2, 4, np.nan, np.nan, 3, np.nan]*10

            })
        ),

        (
            pd.DataFrame({
                "a": ['x', 'x', 'x', 'y', 'x', 'x', 'y']*10,
                "b": [pd.NA, 2, 4, pd.NA, pd.NA, 3, pd.NA]*10

            })
        ),

        (
            pd.DataFrame({
                "a": ['x', 'x', 'x', 'y', 'x', 'x', 'y']*10,
                "b": [None, 2, 4, None, None, 3, None]*10
            })
        ),
    ]
)
def test_CART_one_way_nan(test_data):
    """Test the case where nan always implies-->y and sometimes x--->nan"""
    synth = Synthesiser(random_seed=2)
    fit = synth.fit(test_data).generate(n=100)

    valid = (
        ((fit["a"] == "y") & (fit["b"].isna())) |
        ((fit["a"] == "x") & (fit["b"].isna()+~fit["b"].isna()))
    )

    assert valid.all()
