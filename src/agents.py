from util import get_id
from parameters import DYING_THRESHOLD


class Agent:
    def __init__(
        self,
        init_energy: float,
        nb_offspring: float,
        energy_transmitted: float,
        min_energy: float,
        mutation_probability: float,
        geneneration: bool,
    ):
        self.id: int = get_id()
        self.coords: tuple
        self.generation: int

        # Mutable genes
        self.energy: float = init_energy
        self._nb_offspring: float = nb_offspring
        self.energy_transmited: float = energy_transmitted
        self.min_energy: float = min_energy
        self.mutation_probability: float = mutation_probability

    @property
    def is_alive(self):
        return (self.energy > DYING_THRESHOLD)

    @property
    def birth_threshold(self):
        return (self.min_energy + self.energy_transmited * self.nb_offspring)

    @property
    def nb_offspring(self):
        return round(self._nb_offspring)

    @nb_offspring.setter
    def nb_offspring(self, value):
        self._nb_offspring = value

    @nb_offspring.getter
    def nb_offspring(self):
        return round(self._nb_offspring)
