"""
Diffusion solver for molecules in the assay environment, such as virions,
para, endo, juxta, etc.
"""

import numpy as np


class Homogenous2dDiffusion:
    """Simple diffusion solver for homogenous diffusivity in 2D, aka Heat
    equation"""

    def __init__(self, gamma: float,
                 width: int,
                 height: int,
                 steps_per_step: int,
                 u_init: np.array = None) -> None:
        """
        Args:
            gamma: modified constant for the heat equation.
                   (gamma = alpha * delta_t / delta_x ** 2) where alpha is the 
                   diffusion constant and delta s are discretization resolutions
                   in space and time
            width:
            height:
            steps_per_step: number of time steps for euler computation of equation
                            per model step
            u_init: initial concentration in space. Everywhere zero if not given
        """
        self.u = np.zeros((width, height), dtype=float)
        if u_init is not None:
            self.u = u_init
        self.num_steps = steps_per_step
        self.gamma = gamma
    
    def step(self) -> None:
        for _ in range(self.num_steps):
            u = np.pad(self.u, (1, 1), 'constant', constant_values=0)
            A = u[2:, 1:-1]
            B = u[:-2, 1:-1]
            C = u[1:-1, 2:]
            D = u[1:-1, :-2]
            E = u[1:-1, 1:-1]
            self.u = self.gamma * (A+B+C+D-4*E) + E
            

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # test for diffusion model TODO: write a valid test
    u = np.zeros((500, 500))
    u[200, 300] = 100.0
    diff = Homogenous2dDiffusion(0.2, 100, 100, 30, u)
    for t in range(100):
        plt.imshow(diff.u)
        if t%5 == 0:
            diff.u[np.random.randint(0, 500), np.random.randint(0, 500)] += 10
        diff.step()
        plt.pause(0.001)
        print(t, diff.u.max(), diff.u.sum())