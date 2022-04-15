from environment import Environment
from agents import Agent
from util import get_agent_from_id, get_neighbour_coord, Bidict
from random import choice, shuffle
from parameters import MOVE_COST, IDLE_COST, SIZE
from typing import Dict


class Simulation:
    def __init__(self, parameters) -> None:
        self.size = SIZE
        self.environment: Environment = Environment(self.size, 100, 0.10)
        self.agent_table: Bidict = Bidict((self.size, self.size))
        self.agents: Dict[int, Agent] = dict()

    def update(self):
        self.environment.update()
        self.update_agents()

    def add_agent(self, agent: Agent, coords: tuple):
        if self.agent_table.is_empty(coords):
            agent.coords = coords
            self.agents[agent.id] = agent
            self.agent_table.add_item(coords, agent.id)
        else:
            print("Coordinates occupied, no agent added.")

    def update_agents(self):
        agent_ids = list(self.agents.keys())
        shuffle(agent_ids)
        for agent_id in agent_ids:
            agent = get_agent_from_id(self, agent_id)
            self.move_agent(agent)

    def move_agent(self, agent):
        neighbours = get_neighbour_coord(self, agent.coords)
        if len(neighbours) > 0:
            old_coords = agent.coords  # Old coords
            new_coords = choice(neighbours)  # New coords
            agent.coords = new_coords
            agent.energy -= MOVE_COST
            self.agent_table.swap_coords(old_coords, new_coords)
        else:
            agent.energy -= IDLE_COST

    def draw(self):
        print(" ", end="")
        print("_" * self.size * 2, end="")
        print(" ")

        for i in range(self.size):
            print("|", end="")
            for j in range(self.size):
                if self.agent_table.is_empty((i, j)):
                    print("  ", end="")
                else:
                    print("()", end="")
            print("|")
        print("|", end="")
        print("_" * self.size * 2, end="")
        print("|")
