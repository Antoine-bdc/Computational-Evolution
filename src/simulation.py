from environment import Environment
from agents import Agent, AgentClass
from util import get_agent_from_id, get_empty_neighbour_coord, Bidict
from random import choice, randint, shuffle
from parameters import MOVE_COST, IDLE_COST, SIZE, REGENERATION
from typing import Dict


class Simulation:
    def __init__(self, simulation_id) -> None:
        self.size = SIZE
        self.environment = Environment(self.size, 100, REGENERATION)
        self.agent_table: Bidict = Bidict((self.size, self.size))
        self.agents: Dict[int, Agent] = dict()
        self.simulation_step = 0
        self.simulation_id = simulation_id

    def update(self):
        self.environment.update()
        self.update_agents()
        self.write_data()
        self.simulation_step += 1

    def add_agent(self, agent: Agent, coords: tuple) -> bool:
        if self.agent_table.is_empty(coords):
            agent.coords = coords
            self.agents[agent.id] = agent
            self.agent_table.add_item(coords, agent.id)
            return True
        else:
            return False

    def write_data(self):
        file_name = "step" + str(self.simulation_step)[-8:].zfill(8)
        data_file = open(
            f"data/simulation_{self.simulation_id}/{file_name}", "w",
        )
        for i in range(self.environment.food.shape[0]):
            for j in range(self.environment.food.shape[1]):
                data_file.write(f"{i} ")
                data_file.write(f"{j} ")
                data_file.write(f"{self.environment.food[i, j]} ")
                data_file.write(f"{self.environment.max_food[i, j]} ")
                if not self.agent_table.is_empty((i, j)):
                    agent = get_agent_from_id(
                        self,
                        self.agent_table.table[i, j],
                    )
                    data_file.write(f"{int(agent.type)} ")
                    data_file.write(f"{agent.energy} ")
                    data_file.write(f"{agent._nb_offspring} ")
                    data_file.write(f"{agent.energy_transmited} ")
                    data_file.write(f"{agent.min_energy} ")
                    data_file.write(f"{agent.mutation_probability} ")
                    data_file.write(f"{agent.generation} ")
                else:
                    data_file.write("0 0 0 0 0 0 0")
                data_file.write("\n")
        data_file.close()

    def remove_agent(self, agent):
        self.agents.pop(agent.id)
        self.agent_table.remove_item(agent.coords)
        self.environment.food[agent.coords] += max(0, agent.energy)

    def update_agents(self):
        agent_ids = list(self.agents.keys())
        shuffle(agent_ids)
        for agent_id in agent_ids:
            agent = get_agent_from_id(self, agent_id)
            self.move_agent(agent)
            self.feed_agent(agent)
            self.check_birth(agent)
            self.check_alive(agent)

    def move_agent(self, agent):
        neighbours = get_empty_neighbour_coord(self, agent.coords)
        if len(neighbours) > 0:
            old_coords = agent.coords
            new_coords = choice(neighbours)
            agent.coords = new_coords
            agent.energy -= MOVE_COST[agent.type]
            self.agent_table.swap_coords(old_coords, new_coords)
        else:
            agent.energy -= IDLE_COST[agent.type]

    def feed_agent(self, agent):
        agent.eat(self)

    def check_birth(self, agent):
        neigh_coords = get_empty_neighbour_coord(self, agent.coords)
        shuffle(neigh_coords)
        new_agents = agent.give_birth(len(neigh_coords))
        for child in new_agents:
            self.add_agent(child, neigh_coords.pop(0))

    def check_alive(self, agent):
        if not agent.is_alive:
            self.remove_agent(agent)

    def draw(self):
        print(f"Number of agents = {len(self.agents)}")
        print(" ", end="")
        print("_" * self.size * 2, end="")
        print(" ")

        for i in range(self.size):
            print("|", end="")
            for j in range(self.size):
                if self.agent_table.is_empty((i, j)):
                    print("  ", end="")
                else:
                    print("{}", end="")
            print("|")
        print("|", end="")
        print("_" * self.size * 2, end="")
        print("|")

    def init_agents(self, nb_agents, agent_arguments, agent_type):
        agent_added = 0
        while agent_added < nb_agents:
            if self.add_agent(
                AgentClass[agent_type](*agent_arguments),
                    (randint(0, self.size - 1), randint(0, self.size - 1)),
            ):
                agent_added += 1
