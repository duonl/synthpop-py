"""
This file is a placeholder for another issue:

135-implement-more-integration tests

This test is also present in the current version of that branch, where it failed.

This file should be deleted when issue 135 merges with the develop branch
"""

from synthpop.synthesiser import Synthesiser
from synthpop.methods.cart_synth import CartMethod
from synthpop.methods.copy_synth import CopyMethod
from synthpop.methods.sample_synth import SampleMethod

import pandas as pd
import numpy as np
import pytest

def test_int_bugfix_152():
    df = pd.DataFrame({
    "a": [1, None],
    "b": [0, 0],
    "c": [np.nan, np.nan]
    })

    special_syn_method = {
    "a": SampleMethod(),
    "b": CopyMethod(),
    "c": CartMethod()
    }

    synth = Synthesiser(random_seed=2, special_syn_method=special_syn_method)
    fit = synth.fit(df)