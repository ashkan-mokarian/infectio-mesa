import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


class Matplot:
    """Plots different visualizations using matplotlib.pyplot."""

    def __init__(self, model, State, state_style_dict, figsize=(14, 12)):
        self.fig = plt.figure(figsize=figsize)
        grid = plt.GridSpec(7, 9, wspace=0.1, hspace=0.1)
        self.ax_pos = self.fig.add_subplot(grid[:4, :4])
        self.ax_dif = self.fig.add_subplot(grid[:4, 4:])
        self.ax_colorbar = self.fig.add_subplot(grid[:4, -1], frameon=False)
        # self.ax_lin_radius = self.fig.add_subplot(grid[4, :])
        # self.ax_lin_radius.set_xticks([])
        self.ax_lin_area = self.fig.add_subplot(grid[4, :])
        self.ax_lin_area.set_xticks([])
        self.ax_lin_radial_velocity = self.fig.add_subplot(grid[5, :])
        self.ax_lin_radial_velocity.set_xticks([])
        self.ax_lin_count = self.fig.add_subplot(grid[6, :])
        self.ax_lin_count.set_xlabel("hpi")

        self.ax_pos.set_aspect("auto")
        self.ax_dif.set_aspect("auto")
        self.ax_dif.tick_params(
            axis="y", which="both", left=False, right=False, labelleft=False
        )  # Hide y-ticks for ax_dif
        self.ax_colorbar.tick_params(bottom=False, top=False, left=False, right=False)
        self.ax_colorbar.set_yticklabels([])
        self.ax_colorbar.set_xticklabels([])
        self.colorbar = None
        # ax_dif colorbar normalizer
        # self.difplot_normalizer = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.1, vmax=10)

        self.model = model
        self.State = State
        self.state_style_dict = state_style_dict
        self.steps = []

        # Only keeping I(nfected) states since the graph for S ruins the range
        # for better comparison and not very informative.
        self.num_lists = {k: [] for k in self.State if k is not self.State.S}

        # replacing radius with area
        # Using line plot to show radii mean, max, and min. To avoid keeping
        # the data for previous steps, you need to use animations in pyplot
        # self.radii_mean = []
        # self.radii_max = []
        # self.radii_min = []
        self.area = []

        # For radial velocity
        self.radial_velocity = []

    def update(self, step=None):
        # Initialize three lists for each cell state
        state_lists = {k: [] for k in self.State}
        _ = [state_lists[a.state].append(a) for a in self.model.schedule.agents]
        self.plot_pos(state_lists)
        if step is None:
            step = self.steps[-1] + 1 if self.steps else 0
        self.steps.append(step)
        # self.plot_lin_radius(state_lists[self.State.I])
        self.plot_lin_area()
        self.plot_lin_radius_velocity()
        self.plot_lin_count(state_lists)
        self.plot_particle()

        plt.draw()

    def plot_pos(self, state_lists):
        # visualizing position plots of cells
        self.ax_pos.cla()
        for state in self.State:
            style = self.state_style_dict[state]
            pos = [a.pos for a in state_lists[state]]
            if pos:
                x, y = zip(*pos)
                self.ax_pos.plot(x, y, linestyle="", **style)

    def plot_lin_radius(self, infected_cells):
        """Radius line plots"""
        self.ax_lin_radius.cla()
        self.ax_lin_radius.set_xticks([])
        self.ax_lin_radius.set_ylabel("radius (um) to center of convex")
        if len(infected_cells) >= 3:
            # Get list of infected cells on the boundary of the convex hull
            points = [a.pos for a in infected_cells]
            hull = ConvexHull(points)
            outer_points = [points[i] for i in hull.vertices]
            # Compute center of the convex hull
            center = np.mean(outer_points, axis=0)
            # Compute radii
            radii = (outer_points - center) * 3.1746  # pixel to um
            radii = np.linalg.norm(radii, axis=-1)

        else:
            # no convex hull for 1 or 2 infected cells
            radii = [0]

        self.radii_max.append(np.max(radii))
        self.radii_mean.append(np.mean(radii))
        self.radii_min.append(np.min(radii))

        self.ax_lin_radius.plot(
            [step * (self.model.opt.time_per_step / 3600) for step in self.steps],
            np.array(
                [
                    self.radii_max,
                    self.radii_mean,
                    self.radii_min,
                ]
            ).T,
            label=["max", "mean", "min"],
        )
        self.ax_lin_radius.legend(
            # ["max", "mean", "min"],
            loc="center left",
        )

    def plot_lin_area(self):
        self.ax_lin_area.cla()
        self.ax_lin_area.set_xticks([])
        self.area.append(
            self.model.reporters["plaque_area"].get_areas_in_world_units(
                world_pixel_length=self.model.opt.pixel_length
            )[-1]
        )
        self.ax_lin_area.plot(
            [step * (self.model.opt.time_per_step / 3600) for step in self.steps],
            self.area,
            label="Area (um^2)",
        )
        self.ax_lin_area.legend(loc="center left")

    def plot_lin_radius_velocity(self):
        self.ax_lin_radial_velocity.cla()
        self.ax_lin_radial_velocity.set_xticks([])
        self.radial_velocity.append(
            self.model.reporters[
                "radial_velocity_of_infected_cells"
            ].get_average_radial_velocity_in_world_units(
                world_pixel_length=self.model.opt.pixel_length,
                mins_per_simstep=self.model.opt.time_per_step / 60,
            )
        )
        self.ax_lin_radial_velocity.plot(
            [step * (self.model.opt.time_per_step / 3600) for step in self.steps],
            self.radial_velocity,
            label="radial velocity (um/min)",
        )
        self.ax_lin_radial_velocity.legend(loc="center left")

    def plot_lin_count(self, state_lists):
        """SIR number line plots"""
        self.ax_lin_count.cla()
        steps_in_hrs = [
            step * (self.model.opt.time_per_step / 3600) for step in self.steps
        ]  # xticks are in hpi
        for k, v in self.num_lists.items():
            v.append(len(state_lists[k]))
            self.ax_lin_count.plot(
                steps_in_hrs,
                v,
                "-" + self.state_style_dict[k]["color"],
                label=f"{k.value}: {v[-1]}",
            )
        self.ax_lin_count.set_xlabel("hpi")
        self.ax_lin_count.legend(loc="center left")

    def plot_particle(self):
        if self.model.particle is None:
            return
        self.ax_dif.cla()  # This makes the program run much faster
        data = self.model.particle.u.T

        vmin = np.percentile(data, 5)
        # vmax = np.percentile(data, 95)
        vmax = np.max(data)
        if vmax == 0:
            vmax = 1
        if vmin == 0:
            vmin = 1e-10
        diff_plot = self.ax_dif.imshow(
            self.model.particle.u.T,
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
            origin="lower",
            cmap="turbo",
        )
        if self.colorbar:
            cax = self.colorbar.ax
            cax.clear()
            self.colorbar = self.fig.colorbar(diff_plot, ax=self.ax_colorbar, cax=cax)
        else:
            self.colorbar = self.fig.colorbar(diff_plot, ax=self.ax_colorbar)

    def add_reference_metrics(self, reference_file):

        def read_reference_metrics(
            reference_file,
            reference_colnames={
                "count": {"t": "t", "inf-count-mean": "mean", "inf-count-std": "std"},
                # "radius": {
                #     "t": "t",
                #     "radius-mean(um)": "mean",
                #     "radius-std(um)": "std",
                # },
                "area": {
                    "t": "t",
                    "area-mean(um2)": "mean",
                    "area-std(um2)": "std",
                },
                "radial_velocity": {
                    "t": "t",
                    "radial-velocity-mean(um/min)": "mean",
                    "radial-velocity-std(um/min)": "std",
                },
            },
        ):
            refdf = pd.read_csv(reference_file)
            standard_metrics = {}
            for plot_key, refdf_colnames in reference_colnames.items():
                standard_refdf = refdf.loc[
                    :, [v for v in refdf_colnames.keys()]
                ].rename(columns=refdf_colnames)
                standard_metrics[plot_key] = standard_refdf
            return standard_metrics

        def add_reference_to_plot(ax, standard_df):
            ax.plot(
                standard_df["t"],
                standard_df["mean"],
                label="dataset",
                color="black",
                alpha=0.5,
            )
            ax.fill_between(
                standard_df["t"],
                standard_df["mean"] - standard_df["std"],
                standard_df["mean"] + standard_df["std"],
                alpha=0.2,
                color="black",
            )
            ax.legend(loc="center left")

        standard_reference_metrics = read_reference_metrics(reference_file)
        for plot_key, standard_refdf in standard_reference_metrics.items():
            if plot_key == "count":
                add_reference_to_plot(self.ax_lin_count, standard_refdf)
            elif plot_key == "radius":
                add_reference_to_plot(self.ax_lin_radius, standard_refdf)
            elif plot_key == "area":
                add_reference_to_plot(self.ax_lin_area, standard_refdf)
            elif plot_key == "radial_velocity":
                add_reference_to_plot(self.ax_lin_radial_velocity, standard_refdf)
            else:
                raise ValueError(f"Plot key {plot_key} not recognized.")
