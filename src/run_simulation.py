from simulation import Simulation
from parameters import DEFAULT_AGENT_PARAMETERS, N_ITERATIONS, AgentType
from matplotlib.pyplot import plot, show, legend
from util import plot_data, write_parameters


def run_simulation(simulation, plot_gene_evolution=False):
    simulation_step = 0
    n_prey = []
    n_predator = []
    mutation_probability_list = []
    nb_offspring_list = []
    energy_transmited_list = []
    min_energy_list = []
    while simulation_step < N_ITERATIONS and len(simulation.agents) > 0:
        if (simulation_step % (N_ITERATIONS / 100)) == 0:
            print(f"\r{(1000 * simulation_step / N_ITERATIONS) / 10}%", end="")
        simulation.update()
        # simulation.draw()
        # sleep(0.0002)
        mutation_probability = []
        nb_offspring = []
        energy_transmited = []
        min_energy = []
        n_prey_local = 0
        n_predator_local = 0
        for agent in simulation.agents.values():
            mutation_probability.append(agent.mutation_probability)
            nb_offspring.append(agent._nb_offspring)
            energy_transmited.append(agent.energy_transmited)
            min_energy.append(agent.min_energy)
            if agent.type == AgentType.PREY:
                n_prey_local += 1
            if agent.type == AgentType.PREDATOR:
                n_predator_local += 1
        if n_predator_local == 0:
            break
        n_prey.append(n_prey_local)
        n_predator.append(n_predator_local)

        if len(simulation.agents) > 0:
            mutation_probability_list.append(mutation_probability)
            nb_offspring_list.append(nb_offspring)
            energy_transmited_list.append(energy_transmited)
            min_energy_list.append(min_energy)
        simulation_step += 1

    print("\nSimulation Finished")
    plot(n_prey, label="prey")
    plot(n_predator, label="predator")
    legend()
    show()
    if simulation_step > N_ITERATIONS / 10:
        plot_data(mutation_probability_list, "mutation probability")
        plot_data(nb_offspring_list, "nb offspring")
        plot_data(min_energy_list, "min energy")
        plot_data(energy_transmited_list, "energy transmitted")


if __name__ == "__main__":
    sim_id = write_parameters()
    sim = Simulation(sim_id)
    sim.init_agents(
        50,
        DEFAULT_AGENT_PARAMETERS,
        AgentType.PREY,
    )
    sim.init_agents(
        100,
        [200, 1, 100, 100, 0.2, 0],
        AgentType.PREDATOR,
    )
    run_simulation(sim)
