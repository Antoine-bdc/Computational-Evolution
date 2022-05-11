from enum import IntEnum


class AgentType(IntEnum):
    NONE = 0
    PREY = 1
    PREDATOR = 2
    AGENT = 3


MOVE_COST = {
    AgentType.AGENT: 15,
    AgentType.PREY: 15,
    AgentType.PREDATOR: 4.5,
}
IDLE_COST = {
    AgentType.AGENT: 10,
    AgentType.PREY: 10,
    AgentType.PREDATOR: 3,
}

DYING_THRESHOLD = 10
SIZE = 32
REGENERATION = 0.15
N_ITERATIONS = 10000
DEFAULT_AGENT_PARAMETERS = [100, 1, 100, 100, 0.2, 0]
DEBBUGING = True
