import numpy as np
from matplotlib import pyplot as plt


def rotate_vector(vec, radian):
    return np.array(
        [
            vec[0] * np.cos(radian) - vec[1] * np.sin(radian),
            vec[0] * np.sin(radian) + vec[1] * np.cos(radian),
        ]
    )


def direction_noise(vec, max_noise_radian):
    return rotate_vector(vec, np.random.uniform(-max_noise_radian, max_noise_radian))


def two_slope_production(t, max_value, t1, t2):
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
        plt.plot(T, [two_slope_production(t, 1, 20, 40) for t in T])
        plt.waitforbuttonpress()

    # test_direction_noise()
    test_two_slope_production()
