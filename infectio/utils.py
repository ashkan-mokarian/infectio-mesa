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


if __name__ == "__main__":
    # Testing
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
