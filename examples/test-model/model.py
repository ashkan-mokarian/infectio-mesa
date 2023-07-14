import mesa
import numpy as np

from cell import Cell, State
from infectio.particle import Homogenous2dDiffusion
import options as opt


class TestModel(mesa.Model):
    """
    Test model class for infectio. Handles agent (cell) creation, place them
    randomly, infects one center cell, and scheduling.
    """

    def __init__(self, num_agents, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.space = mesa.space.ContinuousSpace(
            x_max=width, y_max=height, torus=True
        )  # TODO: remove torus, also need to change cell.move()
        # Virions in space, used for molecular diffusion
        self.particle = Homogenous2dDiffusion(
            opt.GAMMA,
            width,
            height,
            opt.virions_diffusion_time_steps_per_each_model_step,
        )
        # Initialize agents randomly
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
        self.particle.step()  # Virion diffusion step
        self.schedule.step()
