"""
Classes to obtain model metrics.
"""

import numpy as np
from scipy.spatial import ConvexHull


# TODO: Add this as default to the model. Instead of using default mesa scheduler
# use lists based on the state of the agents. This is more flexible since many
# of the functionalities depends on the state of the agents.
class StateList:
    def __init__(self, model, states):
        self.model = model
        self.states = states
        self.state_lists = {state: [] for state in self.states}

    def update(self):
        self.state_lists = {state: [] for state in self.states}
        [self.state_lists[a.state].append(a) for a in self.model.schedule.agents]


class StatePos:
    """Tracks x,y positions of the states given at initialization as numpy
    arrays in a state dictionary."""

    def __init__(self, states, state_list: StateList):
        self.states = states
        self.all_states_lists = state_list
        self.state_nppos = {state: np.array([]) for state in self.states}

    def update(self):
        for state in self.states:
            self.state_nppos[state] = np.array(
                [a.pos for a in self.all_states_lists.state_lists[state]]
            )

    def get_xy_np_pos(self, state):
        return self.state_nppos[state]


class Radius:
    """Collects statistics of radii.

    The group of cells need to be specified at each update step. Center is either
    fixed and given at initialization or can be updated at each update step. Also
    plots the radius statistics at each step."""

    def __init__(self, center=None):
        self.center = center
        self.radii_mean = []
        self.radii_std = []
        self.radii_min = []
        self.radii_max = []

    def _update_lists(self, mean, std, min, max):
        self.radii_mean.append(mean)
        self.radii_std.append(std)
        self.radii_min.append(min)
        self.radii_max.append(max)

    def update(self, cell_pos_array, center=None):
        if self.center is None:
            assert (
                center is not None
            ), "Center must be specified for Radius reporter not initialized with center"
        if cell_pos_array is None:
            radii = np.array([0])
        else:
            radii = np.linalg.norm(cell_pos_array - self.center, axis=-1)
        self._update_lists(radii.mean(), radii.std(), radii.min(), radii.max())


class Area:
    """ "Keeps track of the area of the Plaque.

    Using Area as a better metric alternative to radius."""

    def __init__(self) -> None:
        self.area = []

    def update(self, cell_pos_array):
        if cell_pos_array is None or cell_pos_array.shape[0] < 3:
            area = 0
        else:
            area = ConvexHull(cell_pos_array).volume
        self.area.append(area)

    def get_areas(self):
        return self.area if self.area else [0]

    def get_areas_in_world_units(self, world_pixel_length):
        return [a * world_pixel_length**2 for a in self.get_areas()]


class RadialVelocity:
    """Keeps track of the radial velocity of the infected cells.

    Radial velocity is maximum radial component of the trajectory of the infected.
    This class stores the maximum radial for each index."""

    # TODO: see if you can get a reference to the state list at initialization
    # and not pass the list each time in update method
    def __init__(self, center):
        self.center = center
        self.radial_velocity = {}

    def update(self, cell_list):
        for cell in cell_list:
            uid = cell.unique_id
            pos = cell.pos
            if uid not in self.radial_velocity.keys():
                self.radial_velocity[uid] = {
                    "max_rad_velocity": -np.inf,
                    "last_pos": pos,
                }
            else:
                last_pos = self.radial_velocity[uid]["last_pos"]
                rhat = (last_pos - self.center) / np.linalg.norm(last_pos - self.center)
                radial_velocity = np.dot(rhat, pos - last_pos)
                self.radial_velocity[uid]["max_rad_velocity"] = max(
                    radial_velocity, self.radial_velocity[uid]["max_rad_velocity"]
                )

    def average_radial_velocity(self):
        """Average radial velocity of all tracked cells computed based on
        backward difference of simulation steps (normalize accordingly when using.)"""
        rad_vels = [
            self.radial_velocity[uid]["max_rad_velocity"]
            for uid in self.radial_velocity.keys()
            if self.radial_velocity[uid]["max_rad_velocity"] > -np.inf
        ]
        if len(rad_vels) == 0:
            return np.nan
        else:
            return np.nanmean(rad_vels)

    def get_average_radial_velocity_in_world_units(
        self, world_pixel_length, mins_per_simstep
    ):
        return self.average_radial_velocity() * world_pixel_length / mins_per_simstep
