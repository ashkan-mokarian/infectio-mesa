import numpy as np

PARA_PRODUCE_RATE = 1
GRADIENT_SPEED = 1
RW_SPEED = 0.1
GRADIENT_DIRECTION_NOISE_MAX_RADIAN = np.pi / 32
# Heat Diffusion Constants
# all the coefficient used for homogenous diffusion in one.
# \gamma = alpha * delta_t / delta_x ** 2 where alpha is the diffusion constant
GAMMA = 0.2
particle_diffusion_time_steps_per_each_model_step = 30
# 48 hrs (duration of plaque experiments in dataset) * 6 (image sampling every
# 10 minutes) ~= 300 images in total, almost half of the dataset duration which
# where taken from 24h to 48h post infection.
N_SIM_STEPS = 300
