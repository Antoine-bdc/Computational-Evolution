from util import write_parameters, generate_folder
from simulation import Simulation


if __name__ == "__main__":
    folder = generate_folder()
    write_parameters(folder)
    Simulation()
    print(f"End of simulation. \nSimulation data saved in {folder}")
