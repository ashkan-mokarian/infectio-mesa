import mesa
import numpy as np

from cell import Cell, State
from infectio.particle import Homogenous2dDiffusion

# Heat Diffusion Constants
# all the coefficient used for homogenous diffusion in one.
# \gamma = alpha * delta_t / delta_x ** 2 where alpha is the diffusion constant
GAMMA = 0.2
particle_diffusion_time_steps_per_each_model_step = 30


class Model(mesa.Model):
    """
    Test model class for infectio. Handles agent (cell) creation, place them
    randomly, infects one center cell, and scheduling.
    """

    def __init__(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents

        # By having time_infected property for each cell, we don't need to have
        # Multiple schedulers for each state. time_infected also becomes handy
        # For other computations
        # self.schedule = {state: mesa.time.SimultaneousActivation(self)
        #                  for state in CellState}
        self.schedule = mesa.time.SimultaneousActivation(self)

        self.space = mesa.space.ContinuousSpace(
            x_max=width, y_max=height, torus=True
        )  # TODO: remove torus, also need to change cell.move()

        # Virions in space, used for molecular diffusion
        self.particle = Homogenous2dDiffusion(
            GAMMA, width, height, particle_diffusion_time_steps_per_each_model_step
        )

        for i in range(self.num_agents - 1):
            x = self.random.uniform(0, self.space.x_max)
            y = self.random.uniform(0, self.space.y_max)
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
        self.particle.step()
        self.schedule.step()
