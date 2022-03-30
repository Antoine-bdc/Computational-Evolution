from util import fetch_parameters
from simulation import Simulation

if __name__ == "__main__":
    parameters = fetch_parameters("test.txt")
    Simulation(parameters)
    print("End of simulation")
