import pytest

from synthpop.reproducibility import RandomStateManager


@pytest.fixture(autouse=True, scope="function")
def control_random_state_manager(request):
    RandomStateManager.set_root_seed(0)
