import mesa
import numpy as np

from infectio.diffusion_solver import Homogenous2dDiffusion
from infectio.reporters import StateList, StatePos, RadialVelocity, Area
from infectio.utils import get_random_poisson_xy_numbers, get_random_einstein_xy_numbers
from infectio.diffusion_solver import create_diffusion_solver

from cell import Cell, State


class Model(mesa.Model):
    """
    Gradient model class for infectio. Handles agent (cell) creation, place them
    randomly, infects one center cell, add particles, and adds relevant reporters.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.width = opt.width
        self.height = opt.height
        width = self.width
        height = self.height

        # num cells calculated by cell density values. cell density is a random
        # variable and we sample it from a gaussian distribution with known mean
        # and std values from config
        sampled_density = np.random.normal(
            loc=self.opt.cell_density_mean, scale=self.opt.cell_density_std
        )
        self.num_agents = int(
            sampled_density
            * (self.opt.width * self.opt.height * (self.opt.pixel_length) ** 2)
        )

        # By having time_infected property for each cell, we don't need to have
        # Multiple schedulers for each state. time_infected also becomes handy
        # For other computations
        self.schedule = mesa.time.SimultaneousActivation(self)

        self.space = mesa.space.ContinuousSpace(
            x_max=width, y_max=height, torus=True
        )  # TODO: remove torus, also need to change cell.move()

        self.particles = {}

        if opt.enable_vgf:
            # \gamma = alpha * delta_t / delta_x ** 2
            diffusion_delta_t = opt.time_per_step / opt.vgf_diff_steps
            GAMMA = opt.vgf_alpha * diffusion_delta_t / (opt.pixel_length**2)
            self.particles["vgf"] = Homogenous2dDiffusion(
                GAMMA,
                width,
                height,
                opt.vgf_diff_steps,
            )

        if opt.enable_f11:
            diffusion_delta_t = opt.time_per_step / opt.f11_diff_steps
            # GAMMA = opt.f11_alpha * diffusion_delta_t / (opt.pixel_length**2)
            # self.particles["f11"] = Homogenous2dDiffusion(
            #     GAMMA,
            #     width,
            #     height,
            #     opt.f11_diff_steps,
            # )
            self.particle = create_diffusion_solver(
                "cpu",
                width=width,
                height=height,
                steps_per_step=opt.diff_steps,
                u_init=None,
                alpha=opt.alpha,
                dx=opt.pixel_length,
                dt=diffusion_delta_t,
            )

        state_lists = StateList(self, State)
        xypos = StatePos([State.I], state_lists)
        # radius2infcenter = Radius(center=np.array([width / 2, height / 2]))
        plaque_area = Area()
        radial_velocity_of_infected_cells = RadialVelocity(
            center=np.array([width / 2, height / 2]),
        )
        self.reporters = {
            "state_lists": state_lists,
            "xypos": xypos,
            # "radius2infcenter": radius2infcenter,
            "plaque_area": plaque_area,
            "radial_velocity_of_infected_cells": radial_velocity_of_infected_cells,
        }

        # Initialize agents randomly

        if self.opt.initial_random_placement_method == "poisson":
            random_xy_points = get_random_poisson_xy_numbers(
                self.num_agents - 1, 0, self.space.x_max
            )
        elif self.opt.initial_random_placement_method == "einstein":
            random_xy_points = get_random_einstein_xy_numbers(
                self.num_agents - 1,
                0,
                self.space.x_max,
                0,
                self.space.y_max,
                self.opt.initial_random_placement_einstein_factor,
            )
        else:
            raise ValueError(
                "Invalid value for initial_random_placement_einstein_factor (current one is ` "
                + self.opt.initial_random_placement_einstein_factor
                + " `). Change in config file."
            )

        for i in range(self.num_agents - 1):
            x = random_xy_points[i, 0]
            y = random_xy_points[i, 1]
            agent = Cell(i, self)
            self.schedule.add(agent)
            self.space.place_agent(agent, (x, y))

        # put an infected cell in the middle
        agent = Cell(i + 1, self)
        agent.infect_cell()
        self.space.place_agent(agent, (width / 2, height / 2))
        self.schedule.add(agent)

        self.first_inf_cell = agent.unique_id

        self.running = True

    def step(self):
        """
        A model step. Used for collecting data and advancing the schedule
        """
        # Step all active particles
        for p in self.particles.values():
            p.step()
        self.schedule.step()
        self.reporters["state_lists"].update()
        self.reporters["xypos"].update()
        # self.reporters["radius2infcenter"].update(
        #     cell_pos_array=self.reporters["xypos"].get_xy_np_pos(state=State.I)
        # )
        self.reporters["plaque_area"].update(
            cell_pos_array=self.reporters["xypos"].get_xy_np_pos(state=State.I)
        ),
        self.reporters["radial_velocity_of_infected_cells"].update(
            cell_list=self.reporters["state_lists"].state_lists[State.I]
        )

    def save_step(self, frame):
        """
        Only saving position of cells since metrics can be computed from that.
        """
        frame_data = []

        infected_cells = self.reporters["state_lists"].state_lists[State.I]
        for ic in infected_cells:
            frame_data.append(
                [
                    frame,
                    ic.unique_id,
                    "{:.2f}".format(ic.pos[0]),
                    "{:.2f}".format(ic.pos[1]),
                ]
            )
        return frame_data
