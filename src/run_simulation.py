from time import sleep
from random import randint
from simulation import Simulation
from agents import Agent
from parameters import SIZE
from matplotlib.pyplot import plot, show
from numpy import mean


def run_simulation(simulation):
    simulation_step = 0
    n_agents = []
    mutation_probability_list = []
    nb_offspring_list = []
    energy_transmited_list = []
    min_energy_list = []
    while len(simulation.agents) > 0 and simulation_step < 10000:
        simulation.update()
        simulation.draw()
        sleep(0.0002)
        mutation_probability = []
        nb_offspring = []
        energy_transmited = []
        min_energy = []
        for agent in simulation.agents.values():
            mutation_probability.append(agent.mutation_probability)
            nb_offspring.append(agent.nb_offspring)
            energy_transmited.append(agent.energy_transmited)
            min_energy.append(agent.min_energy)
        mutation_probability_list.append(mean(mutation_probability))
        nb_offspring_list.append(mean(nb_offspring))
        energy_transmited_list.append(mean(energy_transmited))
        min_energy_list.append(mean(min_energy))
        simulation_step += 1
        n_agents.append(len(simulation.agents))
    print(simulation_step)
    plot(n_agents)
    show()
    plot(mutation_probability_list)
    show()
    plot(nb_offspring_list)
    show()
    plot(min_energy_list)
    show()
    plot(energy_transmited_list)
    show()


if __name__ == "__main__":
    sim = Simulation()
    for i in range(10):
        sim.add_agent(Agent(100, 1, 100, 100, 0.2, 0),
                      (randint(0, SIZE - 1), randint(0, SIZE - 1)),
                      )
    run_simulation(sim)
