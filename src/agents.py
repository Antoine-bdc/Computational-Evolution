from util import get_id, mutate_value
from parameters import DYING_THRESHOLD


class Agent:
    def __init__(
        self,
        init_energy: float,
        nb_offspring: float,
        energy_transmitted: float,
        min_energy: float,
        mutation_probability: float,
        generation: bool,
    ):
        self.id: int = get_id()
        self.coords: tuple
        self.generation: int = generation
        self.energy: float = init_energy

        # Mutable genes
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
        return min(4, round(self._nb_offspring))

    @nb_offspring.setter
    def nb_offspring(self, value):
        self._nb_offspring = value

    @nb_offspring.getter
    def nb_offspring(self):
        return round(self._nb_offspring)

    def eat(self, food):
        self.energy += food

    def mutated_gene(self, gene):
        return mutate_value(gene, self.mutation_probability)

    def give_birth(self, neigh_space):
        if self.birth_threshold < self.energy:
            max_children = min(neigh_space, self.nb_offspring)
            new_agents = []
            for i in range(max_children):
                new_agents.append(Agent(
                    self.energy_transmited,
                    self.mutated_gene(self._nb_offspring),
                    self.mutated_gene(self.energy_transmited),
                    self.mutated_gene(self.min_energy),
                    self.mutated_gene(self.mutation_probability),
                    # min(1, self.mutated_gene(self.mutation_probability)),
                    self.generation + 1,
                ))
                self.energy -= self.energy_transmited
            return new_agents
        else:
            return []
