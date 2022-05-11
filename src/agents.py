from util import get_id, get_neighbouring_agents, mutate_value
from parameters import DYING_THRESHOLD, AgentType


class Agent:
    def __init__(
        self,
        init_energy: float,
        nb_offspring: float,
        energy_transmitted: float,
        min_energy: float,
        mutation_probability: float,
        generation: bool,
        type=AgentType.AGENT,
    ):
        self.id: int = get_id()
        self.coords: tuple
        self.generation: int = generation
        self.energy: float = init_energy
        self.type = type

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

    def eat(self, sim):
        i, j = self.coords
        self.energy += sim.environment.food[i, j]
        sim.environment.food[i, j] = 0

    def mutated_gene(self, gene):
        return max(0, mutate_value(gene, self.mutation_probability))

    def give_birth(self, neigh_space):
        if self.energy > self.birth_threshold:
            max_children = min(neigh_space, self.nb_offspring)
            new_agents = []

            if self.type is AgentType.PREDATOR:
                agent_class = Predator
            elif self.type is AgentType.PREY:
                agent_class = Prey
            else:
                agent_class = Agent

            for i in range(max_children):
                new_agents.append(
                    agent_class(
                        self.energy_transmited,
                        self.mutated_gene(self._nb_offspring),
                        self.mutated_gene(self.energy_transmited),
                        self.mutated_gene(self.min_energy),
                        self.mutated_gene(self.mutation_probability),
                        self.generation + 1,
                    )
                )
                self.energy -= self.energy_transmited
            return new_agents
        else:
            return []


class Prey(Agent):
    def __init__(
            self,
            init_energy: float,
            nb_offspring: float,
            energy_transmitted: float,
            min_energy: float,
            mutation_probability: float,
            generation: bool,
    ):
        super().__init__(
            init_energy,
            nb_offspring,
            energy_transmitted,
            min_energy,
            mutation_probability,
            generation,
            AgentType.PREY,
        )

    def eat(self, sim):
        i, j = self.coords
        self.energy += sim.environment.food[i, j]
        sim.environment.food[i, j] = 0


class Predator(Agent):
    def __init__(
            self,
            init_energy: float,
            nb_offspring: float,
            energy_transmitted: float,
            min_energy: float,
            mutation_probability: float,
            generation: bool,
    ):
        super().__init__(
            init_energy,
            nb_offspring,
            energy_transmitted,
            min_energy,
            mutation_probability,
            generation,
            AgentType.PREDATOR,
        )

    def eat(self, sim):
        prey = get_neighbouring_agents(sim, self.coords)
        for p in prey:
            if p.type == AgentType.PREY:
                if self.energy > p.energy:
                    self.energy += p.energy
                    p.energy = 0
                    return None


AgentClass = {
    AgentType.PREY: Prey,
    AgentType.PREDATOR: Predator,
    AgentType.AGENT: Agent,
    AgentType.NONE: None,
}
