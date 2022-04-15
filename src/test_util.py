from unittest import case
from util import Bidict, write_parameters
import pytest
from shutil import rmtree
from parameters import MOVE_COST, IDLE_COST, DYING_THRESHOLD, SIZE


@pytest.fixture
def bidict():
    table = Bidict((8, 8))
    table.add_item((2, 2), 5)
    table.add_item((2, 3), 10)
    table.add_item((5, 5), 50)
    return table


def test_swap_ids(bidict):
    bidict.swap_coords((2, 2), (2, 3))
    assert(bidict.table[(2, 2)] == 10)
    assert(bidict.table[(2, 3)] == 5)
    assert(bidict.table[10] == (2, 2))
    assert(bidict.table[5] == (2, 3))


def test_swap_coords(bidict):
    bidict.swap_ids(50, 5)
    assert(bidict.table[(5, 5)] == 5)
    assert(bidict.table[(2, 2)] == 50)
    assert(bidict.table[50] == (2, 2))
    assert(bidict.table[5] == (5, 5))


def test_write_parameters():
    rmtree("data/test")
    write_parameters("test")
    param_txt = open("data/test/parameters.txt", "r")
    for line in param_txt:
        param = line.split()
        registered_value = int(param[1])
        if param[0] == "MOVE_COST":
            assert(MOVE_COST == registered_value)
        if param[0] == "IDLE_COST":
            assert(IDLE_COST == registered_value)
        if param[0] == "DYING_THRESHOLD":
            assert(DYING_THRESHOLD == registered_value)


test_write_parameters()
