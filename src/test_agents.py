import pytest
from agents import Agent
from parameters import DYING_THRESHOLD


@pytest.fixture
def agent():
    return Agent(100, 2.6, 100, 100, 0.4, 1)


def my_agent():
    ag = Agent(100, 2.6, 100, 100, 0.4, 1)
    print(ag.nb_offspring)
    print(ag._nb_offspring)


def test_is_alive(agent):
    assert(agent.is_alive is (agent.energy > DYING_THRESHOLD))
