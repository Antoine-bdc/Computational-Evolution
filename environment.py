import numpy as np


class Environment:
    def __init__(self, N_size, global_max_food, growth_factor) -> None:
        self.size = N_size
        self.max_food = self.initalize_max_food(global_max_food)
        self.food = np.zeros((N_size, N_size))
        self.growth_factor = growth_factor

    def update(self) -> None:
        self.regenerate_food()

    def initalize_max_food(self, max_food):
        return np.random.randint(
            low=0, high=int(max_food),
            size=(self.size, self.size),
            )

    def regenerate_food(self) -> None:
        for i in range(self.size):
            for j in range(self.size):
                local_max = self.max_food[i, j]
                difference = local_max - self.food[i, j]
                self.food[i, j] += difference * self.growth_factor
