import pandas as pd
import pytest

from synthpop.synthesiser import Synthesiser
from synthpop.reproducibility import RandomStateManager

from tests.integration.data_generated_for_tests import get_test_data_regressor


@pytest.mark.parametrize(
    "seed",
    [7, 9, 14, 17, 28, 32, 35]
)
def test_performance_warning_DataFrame_fragmented(seed):
    X, y = get_test_data_regressor(
        seed=seed, with_cats=True, with_missing_features=True, with_missing_target=True)

    RandomStateManager.set_root_seed([seed])
    obs = pd.DataFrame(X)
    obs["target"] = y

    synth = Synthesiser(random_seed=0)
    synth.fit(obs)
    synth.generate(100)
