import pandas as pd
import numpy as np
import pytest

from synthpop.utility_metrics.spmse import pairwise_spmse

@pytest.mark.parametrize(
    "orig_df, syn_df, expected",
    [
        (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), #Check for empty dataframes
        (pd.DataFrame([[1,2],[3,4]]), pd.DataFrame([[1,2],[3,4]]), pd.DataFrame([[0,0],[0,0]])) #Check for identical dataframes
    ] 
)
def test_pairwise_spmse_input(orig_df, syn_df, expected):

    output = pairwise_spmse(orig_df, syn_df)
    assert output.equals(expected)

def test_pairwise_spmse_shapemismatch():
    orig_df = pd.DataFrame([[1,2],[3,4]])
    syn_df = pd.DataFrame([[1,2,3],[4,5,6]])
    with pytest.raises(ValueError):
        pairwise_spmse(orig_df, syn_df)
    