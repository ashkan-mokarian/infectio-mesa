import sys
sys.path.append('.')

import time
import matplotlib.pyplot as plt
import os

from infectio.visualization.matplot import Matplot

from model import TestModel
from cell import State

MAX_SIM_ITER = 500

if __name__ == '__main__':
    # file_name = "output/plots/plot_{}.png"
    # os.makedirs(file_name.format(0), exist_ok=True)
    start_time = time.perf_counter()
    model = TestModel(2000, 500, 500)
    state_style_dict = {
        State.S: {'color': 'b', 'marker': 'o', 'markersize': 4},
        State.I: {'color': 'g', 'marker': 'o', 'markersize': 4},
        State.R: {'color': 'k', 'marker': 'o', 'markersize': 4}
    }

    plot = Matplot(model, State, state_style_dict)
    

    for t in range(MAX_SIM_ITER):
        print(f'step {t}/{MAX_SIM_ITER} Starting ...')
        plot.update(t)
        # plt.savefig(file_name.format(t))
        model.step()
        plt.pause(0.000001)
    print(f"Elapsed time: {time.perf_counter() - start_time:.3f}")
    plt.waitforbuttonpress()
