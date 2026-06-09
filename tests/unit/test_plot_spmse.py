import pytest
import numpy as np
import pandas as pd
from synthpop.plotting.plot import plot_spmse


@pytest.mark.parametrize(
    "s_pmse",
    [
        (pd.DataFrame({"column1": ["c1"], "column2":[2], 
                    "column3": ["c3"], "column4":[2]})), #Can only be exactly 3 columns
    ],
)
def test_input_errors(s_pmse):
    with pytest.raises(ValueError, match ="should be of shape 3xN"):
        output = plot_spmse(s_pmse,  None, False)

def test_plotting():
    s_pmse = pd.DataFrame({"column1": ["c1", "c1", "c1",
                                "c2", "c2", "c3"], 

                        "column2": ["c1", "c2", "c3",
                                "c2", "c3", "c3"], 

                        "S_pMSE": [0, 473842, 12, 
                                4., 46, 0.]})
    plot_spmse(s_pmse, None, True)