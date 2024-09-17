import mesa
import numpy as np

from infectio.particle import Homogenous2dDiffusion
from infectio.reporters import StateList, StatePos, RadialVelocity, Area

from cell import Cell, State


class Model(mesa.Model):
    """
    Gradient model class for infectio. Handles agent (cell) creation, place them
    randomly, infects one center cell, add particles, and adds relevant reporters.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_agents = opt.num_cells
        self.width = opt.width
        self.height = opt.height
        width = self.width
        height = self.height

        # By having time_infected property for each cell, we don't need to have
        # Multiple schedulers for each state. time_infected also becomes handy
        # For other computations
        self.schedule = mesa.time.SimultaneousActivation(self)

        self.space = mesa.space.ContinuousSpace(
            x_max=width, y_max=height, torus=True
        )  # TODO: remove torus, also need to change cell.move()
        self.particle = None
        if not opt.disable_diffusion:
            # VGF particles in space, used for molecular diffusion
            # \gamma = alpha * delta_t / delta_x ** 2 where alpha is the diffusion constant
            diffusion_delta_t = opt.time_per_step / opt.diff_steps
            GAMMA = opt.alpha * diffusion_delta_t / (opt.pixel_length**2)
            self.particle = Homogenous2dDiffusion(
                GAMMA,
                width,
                height,
                opt.diff_steps,
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

        # Poisson
        # for i in range(self.num_agents - 1):
        #     x = self.random.uniform(0, self.space.x_max)
        #     y = self.random.uniform(0, self.space.y_max)
        #     agent = Cell(i, self)
        #     self.schedule.add(agent)
        #     self.space.place_agent(agent, (x, y))

        # Einstein, uniform placement + normaal dist kick
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, self.space.x_max, int(np.sqrt(self.num_agents)) + 1),
            np.linspace(0, self.space.y_max, int(np.sqrt(self.num_agents)) + 1),
        )
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        # Add small random perturbations
        perturbation = 0.5 * (self.width / int(np.sqrt(self.num_agents)))
        perturbation_x = np.random.normal(0, perturbation, grid_points.shape[0])
        perturbation_y = np.random.normal(0, perturbation, grid_points.shape[0])
        grid_points[:, 0] += perturbation_x
        grid_points[:, 1] += perturbation_y
        for i in range(self.num_agents - 2):
            x = grid_points[i, 0]
            y = grid_points[i, 1]
            agent = Cell(i, self)
            self.schedule.add(agent)
            self.space.place_agent(agent, (x, y))

        # put an infected cell in the middle
        agent = Cell(i + 1, self)
        agent.infect_cell()
        self.space.place_agent(agent, (width / 2, height / 2))
        self.schedule.add(agent)

        self.running = True

    def step(self):
        """
        A model step. Used for collecting data and advancing the schedule
        """
        if self.particle:
            self.particle.step()
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
