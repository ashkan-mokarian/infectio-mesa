"""
Diffusion solver for molecules in the assay environment, such as virions,
para, endo, juxta, etc.
"""

import numpy as np
from scipy import ndimage


class Homogenous2dDiffusion:
    """Simple diffusion solver for homogenous diffusivity in 2D, aka Heat
    equation.

    Uses average kernel for for each step of diffusion.
    """

    def __init__(
        self,
        gamma: float,
        width: int,
        height: int,
        steps_per_step: int,
        u_init: np.array = None,
    ) -> None:
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
        self._grad = None

        # To collect histogram of grad orientations of gradient calls
        self.grad_degrees = []

    def step(self) -> None:
        for _ in range(self.num_steps):
            # Faster implementation of the average convolution method for
            # solving the heat equation.
            u = np.pad(self.u, (1, 1), "constant", constant_values=0)
            A = u[2:, 1:-1]
            B = u[:-2, 1:-1]
            C = u[1:-1, 2:]
            D = u[1:-1, :-2]
            E = u[1:-1, 1:-1]
            self.u = self.gamma * (A + B + C + D - 4 * E) + E
        # Caching gradients since some models might not want to compute gradient
        self._grad = None

    def compute_grad(self):
        # Use Sobel instead of np.gradient
        grad_x, grad_y = ndimage.sobel(self.u, axis=0), ndimage.sobel(self.u, axis=1)
        self._grad = np.stack((grad_x, grad_y), axis=-1)

    def grad(self, pos):
        """Returns the gradient at the pixel position.

        Pos is a tuple of float with the exact position of cell in space.
        However this class has only gradients at pixel positions. Therefore
        the position is converted to pixel position."""
        x, y = int(pos[0]), int(pos[1])
        if self._grad is None:
            self.compute_grad()
        # Collect gradians for debugging
        self.grad_degrees.append(np.degrees(np.arctan2(*self._grad[x, y])))

        return self._grad[x, y]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # visual test for diffusion model (Not working)
    u = np.zeros((400, 500))
    u[250:260, 250:260] = 10.0
    diff = Homogenous2dDiffusion(0.2, 100, 100, 30, u)
    X, Y = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]))
    fig, (ax, ax2) = plt.subplots(2, 1)
    for t in range(10):
        ax.cla()
        ax.imshow(diff.u.T, origin="lower")
        if (t + 1) % 2 == 0:
            diff.u[np.random.randint(0, 400), np.random.randint(0, 500)] += 10
        diff.step()
        # testing gradient implementation
        diff.compute_grad()

        # Plot a quiver plot of the gradient colorcoded by the gradient norm
        grad_norms = np.linalg.norm(diff._grad, axis=-1)
        ax.quiver(
            X,
            Y,
            diff._grad[:, :, 0],
            diff._grad[:, :, 1],
            grad_norms,
            cmap="viridis",
            scale=0.01,
        )

        # Plot arrow for max grad in space
        max_norm_index = np.unravel_index(np.argmax(grad_norms), grad_norms.shape)
        arr = diff.grad(max_norm_index)
        ax.arrow(
            max_norm_index[0],
            max_norm_index[1],
            arr[0],
            arr[1],
            color="red",
            head_starts_at_zero=True,
            width=1,
        )

        print("waiting for button press")

        ax2.hist(diff.grad_degrees, bins=20)
        ok = False
        while not ok:
            ok = plt.waitforbuttonpress()
        plt.pause(0.001)
        print(t, diff.u.max(), diff.u.sum())
        plt.pause(0.001)
