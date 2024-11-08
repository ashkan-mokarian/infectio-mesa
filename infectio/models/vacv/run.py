import os
import datetime
import time
import sys
import csv
import json

import configargparse
from dotenv import load_dotenv
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

    # Save parameters of the model from options
    with open(os.path.join(save_path, "params.json"), "w") as f:
        json.dump(vars(opt), f, indent=4)

    save_data = []
    save_metric_data = []

    start_time = time.perf_counter()
    model = Model(opt=opt)
    state_style_dict = {
        State.S: {"color": "b", "marker": "o", "markersize": 4},
        State.I: {"color": "g", "marker": "o", "markersize": 4},
    }

    # Non-interactive backend to run on server and easier plotting
    if not opt.run_gui:
        plt.switch_backend("Agg")
    plot = Matplot(model, State, state_style_dict)

    print(
        "{:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "# Step",
            "t(hours)",
            "# inf",
            "area(um^2)",
            "rad vel(um/min)",
        )
    )
    for t in range(opt.n_sim_steps):
        t_in_hours = t * (opt.time_per_step / 3600)
        plot.update(t)
        if opt.savesnapshots:
            plt.savefig(plot_fn.format(t))
        model.step()
        # Replacing radii metrics with area
        # mean_radius = model.reporters["radius2infcenter"].radii_mean[-1]
        # mean_radius *= opt.pixel_length
        # max_radius = model.reporters["radius2infcenter"].radii_max[-1]
        # max_radius *= opt.pixel_length
        # area = model.reporters["plaque_area"].area[-1]
        # area *= opt.pixel_length**2
        area = model.reporters["plaque_area"].get_areas_in_world_units(
            world_pixel_length=opt.pixel_length
        )[-1]
        # rad_velocity = model.reporters[
        #     "radial_velocity_of_infected_cells"
        # ].average_radial_velocity()
        # rad_velocity *= opt.pixel_length / (
        #     opt.time_per_step / 60
        # )  # TODO: double check these numbers (gives um/min)
        rad_velocity = model.reporters[
            "radial_velocity_of_infected_cells"
        ].get_average_radial_velocity_in_world_units(
            world_pixel_length=opt.pixel_length,
            mins_per_simstep=opt.time_per_step / 60,
        )
        count_infected = len(model.reporters["state_lists"].state_lists[State.I])
        print("{:<10}".format(f"{t+1:03}/{opt.n_sim_steps}"), end="")
        print(
            f"{t_in_hours:<10.2f}{count_infected:<10}{area:<10.2f}{rad_velocity:<10.2f}"
        )  # 3600 to change seconds to hours.
        save_data.append(model.save_step(t))
        save_metric_data.append([t_in_hours, count_infected, area, rad_velocity])
        if opt.run_gui:
            plt.pause(0.000001)

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
        writer.writerow(["t", "infected-count", "area(um2)", "radial-velocity(um/min)"])
        for frame_data in save_metric_data:
            writer.writerow(frame_data)

    # Add reference metrics to the last plot and save it
    plt.savefig(os.path.join(save_path, "plot_final.png"))

    print(f"Elapsed time: {time.perf_counter() - start_time:.3f}")
    if opt.run_gui:
        plt.waitforbuttonpress()


def get_opts():
    default_config = os.path.join(os.path.dirname(__file__), "default_config.ini")
    p = configargparse.ArgParser(default_config_files=[default_config])

    p.add("-c", is_config_file=True, help="config file path")

    p.add(
        "--save_root",
        type=str,
        help="Path where results should be saved (relative to project root).",
    )
    p.add("--save_name", type=str, help="Project save name. current date if not given")
    p.add("--run_gui", action="store_true", help="show plots.")
    p.add(
        "--savesnapshots",
        action="store_true",
        help="whether to save graphs at every step or not.",
    )
    p.add("--n_sim_steps", type=int)
    p.add(
        "--time_per_step",
        type=float,
        help="time in seconds being simulated in one step.",
    )
    p.add("--pixel_length", type=float, help="length of a pixel in micro meters (um).")
    p.add("--width", type=int, help="width of the simulation in pixels.")
    p.add("--height", type=int, help="height of the simulation in pixels.")
    p.add("--num_cells", type=int, help="number of cells to simulate.")
    p.add(
        "--initial_random_placement_method",
        type=str,
        help="poisson (aka uniform) / einstein (posisson+kick)",
    )
    p.add(
        "--initial_random_placement_einstein_factor",
        type=float,
        help="Multiplicative factor for perturbation (between 0 and 1 is recommended, but anything works.)",
    )
    p.add("--disable_diffusion", action="store_true")
    p.add("--alpha", type=float)
    p.add("--diff_steps", type=int)
    p.add("--randomwalk_speed", type=float)
    p.add("--gradient_speed", type=float)
    p.add("--gradient_direction_noise_max", type=float)
    p.add("--para_produce_max", type=float)
    p.add("--para_produce_t1", type=float)
    p.add("--para_produce_t2", type=float)
    p.add("--c2c_sigmoid_k", type=float)
    p.add(
        "--c2c_sigmoid_t0",
        type=float,
        help="time where c2c inf chance stays 0 after being infected.",
    )
    p.add(
        "--c2c_sigmoid_tmid",
        type=float,
        help="time where c2c infection chance gets 0.5.",
    )
    p.add("--c2c_radius_search", type=float)
    p.add(
        "--reference_file",
        type=str,
        help="path to reference file. Contains mean and std of number of infected cells and radius of plaque across different times (relative to project root).",
    )
    p.add(
        "--plot_verbosity",
        type=int,
        help="0: no reference metrics added to plots. 1: only add mean and std. 2: only add individual experimental metrics. 3: add all.",
    )

    options = p.parse_args()
    print(p.format_values())

    load_dotenv()
    PROJECT_PATH = os.getenv(
        "PROJECT_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    )

    if not options.save_root:
        options.save_root = os.path.abspath(
            os.path.join(default_config, "../../../../output")
        )
        print(f"save_root is empty. setting to: {options.save_root}")
    else:
        options.save_root = os.path.abspath(
            os.path.join(PROJECT_PATH, options.save_root)
        )

    options.reference_file = (
        os.path.abspath(os.path.join(PROJECT_PATH, options.reference_file))
        if options.reference_file
        else None
    )

    return options


if __name__ == "__main__":
    options = get_opts()
    print(options)
    run(options)
    print("Finish!")
