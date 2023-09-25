import numpy as np

PARA_PRODUCE_MAX = 1
PARA_PRODUCE_T1 = 15
PARA_PRODUCE_T2 = 40
GRADIENT_SPEED = 1
RW_SPEED = 0.1
GRADIENT_DIRECTION_NOISE_MAX_RADIAN = np.pi / 3

# particle_diffusion_time_steps_per_each_model_step = 30
particle_diffusion_time_steps_per_each_model_step = 40000
# Heat Diffusion Constants
# all the coefficient used for homogenous diffusion in one.
# let alpha be 16.6E-7 cm2/s (reference: jamboard.page5)
# \gamma = alpha * delta_t / delta_x ** 2 where alpha is the diffusion constant
diffusion_delta_t = 10 * 60 / particle_diffusion_time_steps_per_each_model_step
GAMMA = 16.6e-7 * diffusion_delta_t / (3.1746 * 1e-4) ** 2
# 48 hrs (duration of plaque experiments in dataset) * 6 (image sampling every
# 10 minutes) ~= 300 images in total, almost half of the dataset duration which
# where taken from 24h to 48h post infection.
N_SIM_STEPS = 300

# Search radius for deciding cell-cell infection rate based on number of
# infected cells in the radius
CELL2CELL_INFECTION_RADIUS_SEARCH = 5

# Saving
SAVE_ROOT = "./output/"
SAVE_NAME = None  # If None, then the name will be generated automatically

# Only for debug
particle_diffusion_time_steps_per_each_model_step = 40
