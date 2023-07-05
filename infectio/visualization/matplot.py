import matplotlib.pyplot as plt


class Matplot:
    """Plots different visualizations using matplotlib.pyplot."""

    def __init__(
            self,
            model,
            State,
            state_style_dict,
            figsize=(14, 8)):
        self.fig = plt.figure(figsize=figsize)
        grid = plt.GridSpec(4, 7, wspace=0.01, hspace=0.01)
        self.ax_pos = self.fig.add_subplot(grid[:-1, :3])
        self.ax_dif = self.fig.add_subplot(grid[:-1, 3:])
        self.ax_colorbar = self.fig.add_subplot(grid[:-1, -1], frameon=False)
        self.ax_lin = self.fig.add_subplot(grid[-1, :])

        self.ax_pos.set_aspect('auto')
        self.ax_dif.set_aspect('auto')
        self.colorbar = None

        self.model = model
        self.State = State
        self.state_style_dict = state_style_dict
        self.steps = []
        self.num_lists = {k: [] for k in self.State}
    
    def update(self, step=None):
        # Initialize three lists for each cell state
        state_lists = {k: [] for k in self.State}
        _ = [state_lists[a.state].append(a) for a in self.model.schedule.agents]

        # Pos plot
        self.plot_pos(state_lists)
        # Line plot
        if step is None:
            step = self.steps[-1] if self.steps else 0
        self.steps.append(step)
        self.plot_lines(state_lists)
        # Particle plot
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
                self.ax_pos.plot(x, y, linestyle='', **style)

    def plot_lines(self, state_lists):
        """SIR number line plots"""
        self.ax_lin.cla()
        for k, v in self.num_lists.items():
            v.append(len(state_lists[k]))
            self.ax_lin.plot(self.steps, v,
                             '-' + self.state_style_dict[k]['color'],
                             label=f'{k.value}: {v[-1]}')
        self.ax_lin.legend(loc='center left')
    
    def plot_particle(self):
        self.ax_dif.cla() # This makes the program run much faster
        diff_plot = self.ax_dif.imshow(
            self.model.particle.u.T, vmin=0, vmax=1, origin='lower')
        if not self.colorbar:
            self.colorbar = self.fig.colorbar(diff_plot, ax=self.ax_colorbar)