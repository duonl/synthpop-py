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


@pytest.mark.parametrize(
    "missing_value",
    [np.nan, pd.NA, None]
)
def test_missingness_predicts_value(missing_value):
    """A missing should always imply B == 3."""

    test_data = pd.DataFrame({
        "a": [missing_value, 1, missing_value, 2, 3, missing_value] * 20,
        "b": [3, 0, 3, 1, 2, 3] * 20,
    })

    synth = Synthesiser(random_seed=2)
    generated = synth.fit(test_data).generate(n=200)

    rows = generated["a"].isna()

    assert (generated.loc[rows, "b"] == 3).all()


@pytest.mark.parametrize(
    "missing_value",
    [np.nan, pd.NA, None]
)
def test_value_predicts_missingness(missing_value):
    """a == 'x' should always imply b is missing."""

    test_data = pd.DataFrame({
        "a": ["x", "y", "z", "x", "y", "x"] * 20,
        "b": [missing_value, 1, 2, missing_value, 3, missing_value] * 20,
    })

    synth = Synthesiser(random_seed=2)
    generated = synth.fit(test_data).generate(n=200)

    assert generated.loc[generated["a"] == "x", "b"].isna().all()

@pytest.mark.parametrize(
    "missing_value",
    [np.nan, pd.NA, None]
)
def test_joint_missingness_pattern(missing_value):
    """Missing values should occur together."""

    test_data = pd.DataFrame({
        "a": [missing_value, 1, missing_value, 2] * 30,
        "b": [missing_value, 10, missing_value, 20] * 30,
        "c": [5, 6, 7, 8] * 30,
    })

    synth = Synthesiser(random_seed=2)
    generated = synth.fit(test_data).generate(n=200)

    assert (generated["a"].isna() == generated["b"].isna()).all()