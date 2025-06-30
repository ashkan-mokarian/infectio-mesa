"""
Diffusion solver for molecules in the assay environment, such as virions,
para, endo, juxta, etc.
"""

import numpy as np
import argparse
import time
from scipy import ndimage

# Attempt to import CuPy and FFT
try:
    import cupy as cp
    import cupyx.scipy.fft as cufft

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class Homogenous2dDiffusion:
    def __init__(self, gamma, width, height, steps_per_step, u_init=None):
        self.u = np.zeros((width, height), dtype=float)
        if u_init is not None:
            self.u = u_init
        self.num_steps = steps_per_step
        self.gamma = gamma
        self._grad = None

    def step(self):
        for _ in range(self.num_steps):
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
        grad_x = ndimage.sobel(self.u, axis=0)
        grad_y = ndimage.sobel(self.u, axis=1)
        self._grad = np.stack((grad_x, grad_y), axis=-1)

    def grad(self, pos):
        """Returns the gradient at the pixel position.

        Pos is a tuple of float with the exact position of cell in space.
        However this class has only gradients at pixel positions. Therefore
        the position is converted to pixel position."""
        x, y = int(pos[0]), int(pos[1])
        if self._grad is None:
            self.compute_grad()
        return self._grad[x, y]


class CudaDiffusionFFT2D:
    def __init__(self, alpha, width, height, steps_per_step, dx, dt, u_init=None):
        self.alpha = alpha
        self.dx = dx
        self.dt = dt
        self.num_steps = steps_per_step
        self.Nx, self.Ny = width, height
        self.u = cp.zeros((self.Nx, self.Ny), dtype=cp.float32)
        if u_init is not None:
            self.u = cp.array(u_init, dtype=cp.float32)

        kx = cp.fft.fftfreq(self.Nx, d=dx) * 2 * cp.pi
        ky = cp.fft.fftfreq(self.Ny, d=dx) * 2 * cp.pi
        self.KX, self.KY = cp.meshgrid(kx, ky, indexing="ij")
        self.k2 = self.KX**2 + self.KY**2
        self.kernel = cp.exp(-self.alpha * self.k2 * dt)

        self._grad = None
        self.grad_degrees = []

    def update_u(self, u_np):
        self.u = cp.array(u_np, dtype=cp.float32)

    def step(self):
        for _ in range(self.num_steps):
            u_hat = cufft.fft2(self.u)
            u_hat *= self.kernel
            self.u = cp.real(cufft.ifft2(u_hat))
        self._grad = None

    def compute_grad(self):
        from scipy import ndimage

        u_cpu = cp.asnumpy(self.u)
        grad_x = ndimage.sobel(u_cpu, axis=0)
        grad_y = ndimage.sobel(u_cpu, axis=1)
        self._grad = np.stack((grad_x, grad_y), axis=-1)

    def grad(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if self._grad is None:
            self.compute_grad()
        self.grad_degrees.append(np.degrees(np.arctan2(*self._grad[x, y])))
        return self._grad[x, y]

    def get_result(self):
        return cp.asnumpy(self.u)


def create_diffusion_solver(
    method: str,
    width: int,
    height: int,
    steps_per_step: int,
    u_init: np.ndarray,
    alpha: float = None,
    dx: float = None,
    dt: float = None,
):
    if method == "gpu":
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU version requested but CuPy is not installed.")
        if None in [alpha, dx, dt]:
            raise ValueError("GPU version requires alpha, dx, dt.")
        return CudaDiffusionFFT2D(alpha, width, height, steps_per_step, dx, dt, u_init)
    else:
        if None in [alpha, dx, dt]:
            raise ValueError("CPU version requires (alpha, dx, dt).")
        gamma = alpha * dt / dx**2
        return Homogenous2dDiffusion(gamma, width, height, steps_per_step, u_init)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    width, height = 128, 128
    dx, dt = 1.0, 1.0
    alpha = 0.01
    gamma = alpha * dt / dx**2
    u0 = np.zeros((width, height), dtype=np.float32)
    u0[width // 2, height // 2] = 1.0

    solver = create_diffusion_solver(
        method=args.method,
        width=width,
        height=height,
        steps_per_step=args.steps,
        u_init=u0,
        alpha=alpha,
        dx=dx,
        dt=dt,
    )

    print(f"Running {args.method.upper()} solver for {args.steps} steps...")
    t0 = time.time()
    solver.step()
    t1 = time.time()

    result = solver.u if args.method == "cpu" else solver.get_result()
    print(f"Runtime: {t1 - t0:.3f} s")


if __name__ == "__main__":
    main()
