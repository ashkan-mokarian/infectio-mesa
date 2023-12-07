import os
import datetime
import time
import sys
import csv

import configargparse
import matplotlib.pyplot as plt

sys.path.append(".")

from infectio.visualization.matplot_gui import Matplot
from infectio.utils import get_save_path

from model import Model
from cell import State


def run(opt):
    if not opt.save_name:
        opt.save_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = os.path.join(opt.save_root, opt.save_name)
    plot_path = os.path.join(save_path, "plots")
    os.makedirs(plot_path)
    plot_fn = os.path.join(plot_path, "plot{}.png")

    save_data = []
    save_metric_data = []

    start_time = time.perf_counter()
    model = Model(2500, 600, 600, opt=opt)
    state_style_dict = {
        State.S: {"color": "b", "marker": "o", "markersize": 4},
        State.I: {"color": "g", "marker": "o", "markersize": 4},
    }

    # Non-interactive backend to run on server and easier plotting
    if not opt.run_gui:
        plt.switch_backend("Agg")
    plot = Matplot(model, State, state_style_dict)

    print(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "# Step", "t", "# inf", "avg radius", "max radius", "rad vel"
        )
    )
    for t in range(opt.n_sim_steps):
        plot.update(t)
        plt.savefig(plot_fn.format(t))
        model.step()
        mean_radius = model.reporters["radius2infcenter"].radii_mean[-1]
        max_radius = model.reporters["radius2infcenter"].radii_max[-1]
        rad_velocity = model.reporters[
            "radial_velocity_of_infected_cells"
        ].average_radial_velocity()
        count_infected = len(model.reporters["state_lists"].state_lists[State.I])
        print("{:<10}".format(f"{t+1:03}/{opt.n_sim_steps}"), end="")
        print(
            f"{t * (1/6):<10.2f}{count_infected:<10}{mean_radius:<10.2f}{max_radius:<10.2f}{rad_velocity:<10.2f}"
        )
        save_data.append(model.save_step(t))
        save_metric_data.append(
            [t * (1 / 6), count_infected, mean_radius, max_radius, rad_velocity]
        )
        if opt.run_gui:
            plt.pause(0.000001)
    print(f"Elapsed time: {time.perf_counter() - start_time:.3f}")

    # Save data
    with open(os.path.join(save_path, "pos.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["Frame", "CellID", "PosX", "PosY"])
        # Write data for each frame
        for frame_data in save_data:
            writer.writerows(frame_data)
    with open(os.path.join(save_path, "metric.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["t", "Infected_Count", "Mean_Radius", "Max_Radius", "Rad_Velocity"]
        )
        for frame_data in save_metric_data:
            writer.writerow(frame_data)

    if opt.run_gui:
        plt.waitforbuttonpress()


def get_opts():
    default_config = os.path.join(os.path.dirname(__file__), "default_config.ini")
    p = configargparse.ArgParser(default_config_files=[default_config])

    p.add("-c", is_config_file=True, help="config file path")

    p.add("--save_root", type=str, help="Parent directory of results path.")
    p.add("--save_name", type=str, help="Project save name. current date if not given")
    p.add("--run_gui", action="store_true", help="show plots.")
    p.add("--n_sim_steps", type=int)
    p.add("--alpha", type=float)
    p.add("--diff_steps", type=int)
    p.add("--randomwalk_speed", type=float)
    p.add("--gradient_speed", type=float)
    p.add("--gradient_direction_noise_max", type=float)
    p.add("--para_produce_max", type=float)
    p.add("--para_produce_t1", type=float)
    p.add("--para_produce_t2", type=float)
    p.add("--c2c_sigmoid_k", type=float)
    p.add("--c2c_sigmoid_x0", type=float)
    p.add("--c2c_radius_search", type=float)

    options = p.parse_args()
    print(p.format_values())

    if not options.save_root:
        options.save_root = os.path.abspath(
            os.path.join(default_config, "../../../../output")
        )
        print(f"save_root is empty. setting to: {options.save_root}")

    return options


if __name__ == "__main__":
    options = get_opts()
    run(options)
    print("Finish!")
