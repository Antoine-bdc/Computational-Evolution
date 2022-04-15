import pytest
from environment import Environment


@pytest.fixture
def environment():
    return Environment(N_size=8, global_max_food=100, growth_factor=0.5)


def test_max_food_limit(environment):
    for i in range(100):
        environment.update()
    for i in range(environment.size):
        for j in range(environment.size):
            assert(environment.food[i, j] <= environment.max_food[i, j])
