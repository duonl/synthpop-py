import pytest
from synthpop.reproducibility import RandomStateManager

seeds = [0,7,100]
@pytest.fixture(autouse=True,params=seeds,scope="session")
def control_random_state_manager(request):
    seed = request.param
    RandomStateManager.set_root_seed(seed)
