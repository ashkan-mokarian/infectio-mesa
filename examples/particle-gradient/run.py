import sys

sys.path.append(".")

import time
import matplotlib.pyplot as plt
import os

from infectio.visualization.matplot import Matplot

from model import Model
from cell import State
import options as opt


if __name__ == "__main__":
    # file_name = "output/plots/plot_{}.png"
    # os.makedirs(file_name.format(0), exist_ok=True)
    start_time = time.perf_counter()
    model = Model(2000, 500, 500)
    state_style_dict = {
        State.S: {"color": "b", "marker": "o", "markersize": 4},
        State.I: {"color": "g", "marker": "o", "markersize": 4},
    }

    plot = Matplot(model, State, state_style_dict)

    for t in range(opt.N_SIM_STEPS):
        print(f"step {t}/{opt.N_SIM_STEPS} Starting ...")
        plot.update(t)
        # plt.savefig(file_name.format(t))
        model.step()
        plt.pause(0.000001)
    print(f"Elapsed time: {time.perf_counter() - start_time:.3f}")
    plt.waitforbuttonpress()
