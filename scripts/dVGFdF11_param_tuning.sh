#!/bin/bash

# Define the parameter values
randomwalk_speeds=(0.01 0.1 0.5 1.0 5 10 50)
c2c_sigmoid_k=(0.1 0.5 1 5 10 50)
c2c_sigmoid_x0=(1 2 5 10 20 50)
c2c_radius_search=(2 5 10 15 20 50)
num_experiments=5

# grid search for parameter identification
for randomwalk_speed in "${randomwalk_speeds[@]}"; do
    # Another inner loop for c2c_sigmoid_k
    for k_value in "${c2c_sigmoid_k[@]}"; do
        # Another inner loop for c2c_sigmoid_x0
        for x0_value in "${c2c_sigmoid_x0[@]}"; do
            # Yet another inner loop for c2c_radius_search
            for radius_value in "${c2c_radius_search[@]}"; do
                for ((experiment=1; experiment<=num_experiments; experiment++)); do
                    random_suffix=$((RANDOM % 10000))
                    save_name="randomwalk_speed=${randomwalk_speed}-grad_speed=${grad_speed}-c2c_k=${k_value}-c2c_x0=${x0_value}-c2c_radius=${radius_value}-exp${random_suffix}"
                    sbatch ./scripts/cpuN1C2_param_identification.sbatch "-c" "./examples/vacv-epithelial/dVGFdF11_default_config.ini" "--save_name" $save_name "--randomwalk_speed" "$randomwalk_speed" "--c2c_sigmoid_k" "$k_value" "--c2c_sigmoid_x0" "$x0_value" "--c2c_radius_search" "$radius_value"
            done
        done
    done
done