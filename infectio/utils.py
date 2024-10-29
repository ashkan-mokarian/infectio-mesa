from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import os


def rotate_vector(vec, radian):
    return np.array(
        [
            vec[0] * np.cos(radian) - vec[1] * np.sin(radian),
            vec[0] * np.sin(radian) + vec[1] * np.cos(radian),
        ]
    )


def direction_noise(vec, max_noise_radian):
    return rotate_vector(vec, np.random.uniform(-max_noise_radian, max_noise_radian))


def two_slope_function(t, max_value, t1, t2):
    """Utility function for two-slope molecule production.

    Args:
        t (float): point to evaluate function at.
        max_value (float): Maximum value.
        t1 (float): time where maximum reached after infection.
        t2 (float): time where production is 0 again.

    Returns:
        float: Value.

    Raises:
        ValueError: If t1 is greater than t2.
        ValueError: If t is negative.
    """
    assert t1 < t2, "t1 must be smaller than t2"
    assert t >= 0, "t must be positive"
    if t < t1:
        return (t / t1) * max_value
    elif t >= t1 and t < t2:
        return max_value * (1 - (t - t1) / (t2 - t1))
    else:
        return 0


def get_save_path(root, name=None):
    """Handy function to create project root directory, useful when running
    multiple jobs in a cluster"""
    if name is None:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(root, name)
    i = 1
    while os.path.exists(save_path):
        save_path = os.path.join(root, name + f"_{i}")
        i += 1
    return save_path


def get_random_poisson_xy_numbers(n, x_min, x_max, y_min, y_max):
    random_xy = np.empty((n, 2))
    random_xy[:, 0] = np.random.uniform(x_min, x_max, size=n)
    random_xy[:, 1] = np.random.uniform(y_min, y_max, size=n)
    return random_xy


def get_random_einstein_xy_numbers(n, x_min, x_max, y_min, y_max, einstein_factor):
    # Einstein, uniform placement + normaal dist kick
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, int(np.sqrt(n)) + 1),
        np.linspace(y_min, y_max, int(np.sqrt(n)) + 1),
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    # Add small random perturbations
    perturbation = einstein_factor * (
        np.sqrt((x_max - x_min) * (y_max - y_min)) / int(np.sqrt(n))
    )
    perturbation_x = np.random.normal(0, perturbation, grid_points.shape[0])
    perturbation_y = np.random.normal(0, perturbation, grid_points.shape[0])
    grid_points[:, 0] += perturbation_x
    grid_points[:, 1] += perturbation_y
    return grid_points


if __name__ == "__main__":
    # Testing

    # Test for direction_noise
    def test_direction_noise():
        vectors = [
            [1, 0],
            [0, 1],
            [1, 1],
            [-1, 1],
            [1, -1],
            [-1, -1],
            [-1, 0],
            [0, -1],
        ] * 50
        vectors = [v / np.linalg.norm(v) for v in vectors]
        orig = [np.degrees(np.arctan2(x, y)) for (x, y) in vectors]
        new_vectors = [direction_noise(v, np.pi / 2) for v in vectors]
        changed = [np.degrees(np.arctan2(x, y)) for (x, y) in new_vectors]
        plt.hist([orig, changed], label=["Original", "Changed"], bins=360)
        plt.waitforbuttonpress()

    # Test for two_slope_production
    def test_two_slope_production():
        T = np.arange(0, 50)
        plt.plot(T, [two_slope_function(t, 1, 20, 40) for t in T])
        plt.waitforbuttonpress()

    # test_direction_noise()
    test_two_slope_production()
