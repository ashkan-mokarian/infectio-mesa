import numpy as np
from enum import Enum

import mesa

from infectio.utils import direction_noise, two_slope_function


# Cell states
class State(Enum):
    S = "Susceptible"
    I = "Infected"


# For now, let's choose a simple sigmoid function for deciding cell-cell infection chance
def cell2cell_infection_chance(num_infected_neighbors, k, x0):
    if num_infected_neighbors == 0:
        return 0.0
    return 1 / (1 + np.exp(-k * (num_infected_neighbors - x0)))


class Cell(mesa.Agent):
    """
    A Cell agent
    """

    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        super().__init__(unique_id, model)

        # Cell state
        self.state = State.S
        self.time_infected = None
        self.opt = model.opt

    def particle_gradient_direction(self):
        """Returns gradient direction of particle"""
        grad = self.model.particle.grad(self.pos)
        return grad

    def move(self):
        """Walk in opposite particle gradient direction of para molecule plus
        RW."""
        dx, dy = self.random.random() - 0.5, self.random.random() - 0.5
        rw = np.array([dx, dy])
        rw /= np.linalg.norm(rw)
        new_pos = self.pos + rw * self.opt.randomwalk_speed
        if self.state is State.I:
            grad_dir = self.particle_gradient_direction()
            norm = np.linalg.norm(grad_dir)
            if norm:
                grad_dir /= norm
                grad_dir = direction_noise(
                    grad_dir, self.opt.gradient_direction_noise_max
                )
                new_pos -= grad_dir * self.opt.gradient_speed
        self.model.space.move_agent(self, new_pos)

    def infect_cell(self):
        assert self.state is State.S, "Only healthy cells can get infected."
        self.state = State.I
        self.time_infected = 0

    def decide_to_infect(self):
        """Randomly infect healthy cells based on proximity to other infected cells."""
        if self.state is not State.S:
            return
        # Simple cell-cell infection only based on numbers
        infected_neighbors = [
            c
            for c in self.model.space.get_neighbors(
                self.pos,
                radius=self.opt.c2c_radius_search,
                include_center=False,
            )
            if c.time_infected
        ]  # c.time_infected checks for both, if cell is
        # infected, and also, if it wasn't decided to
        # get infected in the current step
        # TODO: what distribution to use and what parameters. For now, sigmoid
        # with linear exponentiation
        infection_prob = cell2cell_infection_chance(
            len(infected_neighbors), self.opt.c2c_sigmoid_k, self.opt.c2c_sigmoid_x0
        )
        if infection_prob > self.random.random():
            self.infect_cell()

    def add_para_molecule_during_infection(self):
        x, y = self.pos
        # Two-slope production for infected cells
        self.model.particle.u[int(x), int(y)] += two_slope_function(
            self.time_infected,
            self.opt.para_produce_max,
            self.opt.para_produce_t1,
            self.opt.para_produce_t2,
        )

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        self.decide_to_infect()

    def advance(self) -> None:
        self.move()
        if self.state is State.I:
            self.time_infected += 1
            self.add_para_molecule_during_infection()
