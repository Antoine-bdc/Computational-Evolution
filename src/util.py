from typing import Tuple
from os import mkdir, path
from random import random


id_counter = 1
neg_id = -1


def write_parameters(folder) -> None:
    if path.exists(f"data/{folder}"):
        print("Path already exists. Parameters not written")
        return None
    mkdir(f"data/{folder}")
    data_file = open(f"data/{folder}/parameters.txt", 'w')
    parameters_file = open("src/parameters.py", 'r')
    for line in parameters_file:
        split_line = line.split()
        data_file.write(f"{split_line[0]} {split_line[2]}\n")
    data_file.close()
    parameters_file.close()


def get_neighbour_coord(sim, coords, none_value=0):
    i, j = coords
    neighbour_coords = [
        ((i + 1) % sim.size, j),
        ((i - 1) % sim.size, j),
        (i, (j + 1) % sim.size),
        (i, (j - 1) % sim.size),
    ]
    output = []
    for n in neighbour_coords:
        i, j = n
        if sim.agent_table.is_empty(((i, j))):
            output.append(n)
    return output


def get_id() -> int:
    global id_counter
    id_counter += 1
    return id_counter


def get_neg_id() -> int:
    global neg_id
    neg_id -= 1
    return neg_id


def get_agent_from_id(sim, id):
    return sim.agents[id]


def get_agent_from_coord(sim, coords):
    return get_agent_from_id(sim.agent_table[coords])


def mutate_value(value, mutation_probability):
    return value + value * (random() - 0.5) * mutation_probability


class Bidict:
    def __init__(self, size: Tuple) -> None:
        self.table = dict()
        for i in range(size[0]):
            for j in range(size[1]):
                n_id = get_neg_id()
                self.table[(i, j)] = n_id
                self.table[n_id] = (i, j)

    def add_item(self, coord: Tuple[int, int], id: int) -> None:
        self.table[coord] = id
        self.table[id] = coord

    def remove_item(self, coord: Tuple) -> None:
        n_id = get_neg_id()
        self.table[coord] = n_id
        self.table[n_id] = coord

    def swap_coords(self, coord1: Tuple, coord2: Tuple) -> None:
        id_1 = self.table[coord1]
        id_2 = self.table[coord2]
        self.table[coord1], self.table[coord2] = (id_2, id_1)
        self.table[id_1], self.table[id_2] = (coord2, coord1)

    def swap_ids(self, id_1: Tuple, id_2: Tuple) -> None:
        coord_1 = self.table[id_1]
        coord_2 = self.table[id_2]
        self.table[id_1], self.table[id_2] = (coord_2, coord_1)
        self.table[coord_1], self.table[coord_2] = (id_2, id_1)

    def is_empty(self, coords) -> bool:
        return (self.table[coords] < 0)
