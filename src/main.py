from simulation import Simulation
from parameters import DEFAULT_AGENT_PARAMETERS
from util import write_parameters
from run_simulation import run_simulation


def setup_simulations(nb_simulations, plot_data=True):
    for i in range(nb_simulations):
        sim_id = write_parameters()
        sim = Simulation(sim_id)
        sim.init_agents(
            30,
            DEFAULT_AGENT_PARAMETERS,
        )
        run_simulation(sim)
