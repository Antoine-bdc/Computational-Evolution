# from time import sleep
from random import randint
from simulation import Simulation
from agents import Agent
from parameters import SIZE, N_ITERATIONS
from matplotlib.pyplot import plot, show
from util import mutate_value, plot_data  # , AgentType


def run_simulation(simulation):
    simulation_step = 0
    n_agents = []
    mutation_probability_list = []
    nb_offspring_list = []
    energy_transmited_list = []
    min_energy_list = []
    while simulation_step < N_ITERATIONS and len(simulation.agents) > 0:
        if (simulation_step % (N_ITERATIONS / 100)) == 0:
            print(f"\r{100 * simulation_step / N_ITERATIONS}%", end="")
        simulation.update()
        # simulation.draw()
        # sleep(0.0002)
        mutation_probability = []
        nb_offspring = []
        energy_transmited = []
        min_energy = []
        for agent in simulation.agents.values():
            mutation_probability.append(agent.mutation_probability)
            nb_offspring.append(agent._nb_offspring)
            energy_transmited.append(agent.energy_transmited)
            min_energy.append(agent.min_energy)

        if len(simulation.agents) > 0:
            mutation_probability_list.append(mutation_probability)
            nb_offspring_list.append(nb_offspring)
            energy_transmited_list.append(energy_transmited)
            min_energy_list.append(min_energy)
        n_agents.append(len(simulation.agents))
        simulation_step += 1

    print("\nSimulation Finished")
    plot(n_agents)
    show()
    if simulation_step == N_ITERATIONS:
        plot_data(mutation_probability_list, "mutation probability")
        plot_data(nb_offspring_list, "nb offspring")
        plot_data(min_energy_list, "min energy")
        plot_data(energy_transmited_list, "energy transmitted")


if __name__ == "__main__":
    sim = Simulation()
    for i in range(30):
        mutation_probability = mutate_value(0.2, 0.2)
        n_offspring = mutate_value(1, mutation_probability)
        energy_transmited = mutate_value(100, mutation_probability)
        min_energy = mutate_value(100, mutation_probability)
        sim.add_agent(
            Agent(
                100,
                n_offspring,
                energy_transmited,
                min_energy,
                mutation_probability,
                0,
            ),
            (randint(0, SIZE - 1), randint(0, SIZE - 1)),
        )
    run_simulation(sim)
