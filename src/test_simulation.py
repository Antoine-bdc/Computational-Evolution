import pytest
from time import sleep
from random import randint
from simulation import Simulation
from agents import Agent
from parameters import SIZE


@pytest.fixture
def simulation():
    sim = Simulation(dict())
    for i in range(10):
        sim.add_agent(Agent(100, 1, 100, 100, 0.8),
                      (randint(0, SIZE - 1), randint(0, SIZE - 1))
                      )
    return sim


def run_simulation(simulation):
    for i in range(100):
        simulation.update()
        simulation.draw()
        sleep(0.1)


if __name__ == "__main__":
    sim = Simulation()
    for i in range(10):
        sim.add_agent(Agent(100, 1, 100, 100, 0.8),
                      (randint(0, SIZE - 1), randint(0, SIZE - 1)),
                      0,
                      )
    run_simulation(sim)
