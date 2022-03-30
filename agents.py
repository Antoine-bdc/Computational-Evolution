from util import get_id
from parameters import DYING_THRESHOLD


class Agent:
    def __init__(
        self,
        init_energy,
        nb_offspring,
        energy_transmitted,
        min_energy,
        mutation_probability,
    ):
        self.id = get_id()
        self.coords = None

        # Mutable genes
        self.energy = init_energy
        self.nb_offspring = nb_offspring
        self.energy_transmited = energy_transmitted
        self.min_energy = min_energy
        self.mutation_probability = mutation_probability

    @property
    def is_alive(self):
        return (self.energy > DYING_THRESHOLD)

    @property
    def birth_threshold(self):
        return (self.min_energy + self.energy_transmited * self.nb_offspring)
