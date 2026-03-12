from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import NotFittedError
import numpy as np
import pandas as pd
import pytest 
from synthpop.data_processing.missing_value_handling import ReplaceNoneWithValue


def test_prepare_data_for_fit_numeric_correctness():
    estimator = ReplaceNoneWithValue()