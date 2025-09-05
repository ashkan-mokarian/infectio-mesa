import numpy as np
from enum import Enum

import mesa

from infectio.utils import direction_noise, two_slope_function


# Cell states
class State(Enum):
    S = "Susceptible"
    I = "Infected"


def sigmoid_infectivity_chance(t, t0, k, tmid):
    return (
        0
        if t < t0
        else (1 / (1 + np.exp(-k * (t - tmid))) - 1 / (1 + np.exp(-k * (t0 - tmid))))
        * (
            (1 + np.exp(-k * (t0 - tmid))) / (np.exp(-k * (t0 - tmid)))
        )  # almost the usual sigmoid function, just subtracted such that it gets 0 at t0 and also multiplied such that it goes to 1
    )


def cell2cell_infection_chance_add_sigmoids(
    infected_neighbors, first_inf_cell_t_inf, lag_time, t0, k, tmid
):

    chances = sum(
        [
            sigmoid_infectivity_chance(c.time_infected, t0, k, tmid)
            for c in infected_neighbors
        ]
    )
    if first_inf_cell_t_inf > 0:
        chances += sigmoid_infectivity_chance(
            first_inf_cell_t_inf, t0 + lag_time, k, tmid + lag_time
        )
    return chances


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
        if self.model.particle is None:
            return np.array([0, 0])
        grad = self.model.particle.grad(self.pos)
        return grad

    def move(self):
        """Walk in opposite particle gradient direction of para molecule plus
        RW."""
        dx, dy = self.random.random() - 0.5, self.random.random() - 0.5
        rw_direction = np.array([dx, dy])
        norm = np.linalg.norm(rw_direction)
        if norm:
            rw_direction /= norm

        random_distance = self.random.random() * self.opt.rw_speed_in_pixels_per_step
        randomwalk_vector = rw_direction * random_distance
        new_pos = self.pos + randomwalk_vector

        if self.state is State.I:
            grad_dir = self.particle_gradient_direction()
            norm = np.linalg.norm(grad_dir)
            if norm:
                grad_dir /= norm
                grad_dir = direction_noise(
                    grad_dir, self.opt.gradient_direction_noise_max
                )
                new_pos -= grad_dir * self.opt.gradient_speed_in_pixels_per_step
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
        infected_neighbors = []
        first_inf_cell_t_inf = 0
        for c in self.model.space.get_neighbors(
            self.pos, radius=self.opt.c2c_radius_search_in_pixels, include_center=False
        ):
            if not c.time_infected:
                continue
            if c.unique_id == self.model.first_inf_cell:
                first_inf_cell_t_inf = c.time_infected
            else:
                infected_neighbors.append(c)
        # infected_neighbors = [
        #     c
        #     for c in self.model.space.get_neighbors(
        #         self.pos,
        #         radius=self.opt.c2c_radius_search,
        #         include_center=False,
        #     )
        #     if c.time_infected
        # ]  # c.time_infected checks for both, if cell is
        # infected, and also, if it wasn't decided to
        # get infected in the current step

        infection_prob = cell2cell_infection_chance_add_sigmoids(
            infected_neighbors,
            first_inf_cell_t_inf,
            self.model.opt.first_cell_lag_in_steps,
            self.opt.c2c_sigmoid_t0_in_steps,
            self.opt.c2c_sigmoid_k,
            self.opt.c2c_sigmoid_tmid_in_steps,
        )

        if infection_prob > self.random.random():
            self.infect_cell()

    def add_para_molecule_during_infection(self):
        if self.model.particle is None:
            return
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
