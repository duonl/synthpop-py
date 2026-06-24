import string

import numpy as np
import pandas as pd
import pytest


from synthpop.synthesiser import Synthesiser


from tests.integration.data_generated_for_tests import get_test_data_classifier,get_test_data_regressor,simulate_realistic_dataset_correlations

def combined_regressor_and_classifier_test_data(seed = 10):
    X_reg,y_reg = get_test_data_regressor(seed=seed,with_cats=True,with_missing_features=True,with_missing_target=True)
    X_clas,y_clas = get_test_data_classifier(seed=seed,with_cats=True,with_missing_features=True,with_missing_target=True)

    d_data = {}

    available_columns = set(X_reg.keys()).intersection(set(X_clas.keys()))

    for i,k in enumerate(available_columns):

        if i %2 ==0:
            d_data[k] = X_reg[k]
        else:
            d_data[k] = X_clas[k]

    d_data['y1'] = y_reg
    d_data['y2']= y_clas

    return pd.DataFrame(d_data)



def test_reproducibility_synthesis():

    obs = combined_regressor_and_classifier_test_data()#simulate_realistic_dataset_correlations(n_samples=1000)[0]

    synth = Synthesiser(random_seed=1)
    synth.fit(obs)

    syn1 = synth.generate(2000)
    syn2 = synth.generate(2000)

    assert syn1.equals(syn2), "generating 2 consecutive times did not produce the same synthetic dataset"


    synth2 = Synthesiser(random_seed=1)
    synth2.fit(obs)

    syn3 = synth2.generate(2000)

    for col in syn3.columns:
        #pd.testing.assert_series_equal(syn3[col],syn2[col])

        if col == '9':
            print("stop")

        syn3_is_nan_mask = pd.isna(syn3[col])
        syn2_is_nan_mask = pd.isna(syn2[col])
        assert syn2_is_nan_mask.equals(syn3_is_nan_mask), f"missingness not reproduced for column {col}"
        assert (syn3[col][~syn3_is_nan_mask] == syn2[col][~syn2_is_nan_mask]).all(), f"column {col} not reproduced"

def test_generate_independent_syn_datasets():

    obs = simulate_realistic_dataset_correlations(n_samples=1010)[0]

    synth = Synthesiser(random_seed=0)
    synth.fit(obs)

    syn1 = synth.generate(n=100)
    syn2 = synth.generate(n=100,random_seed=1234)

    assert not syn1.equals(syn2)
