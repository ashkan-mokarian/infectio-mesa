import os
import datetime
import time
import sys
import csv

sys.path.append(".")

import matplotlib.pyplot as plt

from infectio.visualization.matplot_gui import Matplot

from model import Model
from cell import State
import options as opt
import csv

# Non-interactive backend to run on server and easier plotting
plt.switch_backend("Agg")

if __name__ == "__main__":
    if opt.SAVE_NAME is None:
        opt.SAVE_NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = os.path.join(opt.SAVE_ROOT, opt.SAVE_NAME)
    plot_path = os.path.join(save_path, "plots")
    os.makedirs(plot_path)
    plot_fn = os.path.join(plot_path, "plot{}.png")

    save_data = []

    start_time = time.perf_counter()
    model = Model(2500, 600, 600)
    state_style_dict = {
        State.S: {"color": "b", "marker": "o", "markersize": 4},
        State.I: {"color": "g", "marker": "o", "markersize": 4},
    }

    plot = Matplot(model, State, state_style_dict)

    for t in range(opt.N_SIM_STEPS):
        print(f"step {t}/{opt.N_SIM_STEPS} Starting ...")
        plot.update(t)
        plt.savefig(plot_fn.format(t))
        model.step()
        save_data.append(model.save_step(t))
        # plt.pause(0.000001)
    print(f"Elapsed time: {time.perf_counter() - start_time:.3f}")

    # Save data
    with open(os.path.join(save_path, "pos.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["Frame", "CellID", "PosX", "PosY"])
        # Write data for each frame
        for frame_data in save_data:
            writer.writerows(frame_data)

    # Final results
    print(
        f"Average radial velocity of infected cells: "
        f"{model.reporters['radial_velocity_of_infected_cells'].average_radial_velocity():.3f} um/min"
    )

    # plt.waitforbuttonpress()
