# generate_param_list.py

import itertools

# Define your parameter values
randomwalk_speeds = [0.4, 0.5, 0.6]
c2c_sigmoid_k = [0.05, 0.1, 0.25]
c2c_sigmoid_t0 = [6, 12, 24, 36]
c2c_sigmoid_tmid = [24, 36, 48, 64]
c2c_radius_search = [20, 25, 30]
first_cell_lags = [36, 48, 64]

# Generate combinations
combinations = itertools.product(
    randomwalk_speeds,
    c2c_sigmoid_k,
    c2c_sigmoid_t0,
    c2c_sigmoid_tmid,
    c2c_radius_search,
    first_cell_lags,
)

with open("parameter_combinations.txt", "w") as f:
    for comb in combinations:
        (
            randomwalk_speed,
            k,
            t0,
            tmid,
            radius,
            lag,
        ) = comb

        # # Modify the list as you want either here or seperately in the file.
        if tmid < t0:
            continue

        save_name = (
            f"randomwalk_speed={randomwalk_speed}-"
            f"c2c_sigmoid_k={k}-"
            f"c2c_sigmoid_t0={t0}-"
            f"c2c_sigmoid_tmid={tmid}-"
            f"c2c_radius_search={radius}-"
            f"first_cell_lag={lag}"
        )
        f.write(save_name + "\n")
