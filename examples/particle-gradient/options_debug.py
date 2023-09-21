import numpy as np

PARA_PRODUCE_MAX = 1
PARA_PRODUCE_T1 = 15
PARA_PRODUCE_T2 = 40
GRADIENT_SPEED = 1
RW_SPEED = 0.1
GRADIENT_DIRECTION_NOISE_MAX_RADIAN = np.pi / 32
# Heat Diffusion Constants
# all the coefficient used for homogenous diffusion in one.
# \gamma = alpha * delta_t / delta_x ** 2 where alpha is the diffusion constant
GAMMA = 0.2
particle_diffusion_time_steps_per_each_model_step = 3
# 48 hrs (duration of plaque experiments in dataset) * 6 (image sampling every
# 10 minutes) ~= 300 images in total, almost half of the dataset duration which
# where taken from 24h to 48h post infection.
N_SIM_STEPS = 30

# Search radius for deciding cell-cell infection rate based on number of
# infected cells in the radius
CELL2CELL_INFECTION_RADIUS_SEARCH = 5

# Saving params
SAVE_ROOT = "./out/debug/"
SAVE_NAME = None  # If None, will be date-time
