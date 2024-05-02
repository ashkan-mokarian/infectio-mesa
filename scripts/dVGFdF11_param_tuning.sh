#!/bin/bash

# Define the parameter values
randomwalk_speeds=(0.1 0.25 0.5 0.75)
gradient_speeds=(0)
gradient_direction_noise_maxs=(0.1 0.25 0.5 1)
c2c_sigmoid_k=(0.25 0.5 1 2)
c2c_sigmoid_x0=(2 3 4 5)
c2c_radius_search=(3 5 7 9 11)

# grid search for parameter identification
for radomwalk_speed in "${randomwalk_speeds[@]}"; do
    # Inner loop for gradient_speed
    for grad_speed in "${gradient_speeds[@]}"; do
        # Inner loop for gradient_direction_noise_max
        for gradient_direction_noise_max in "${gradient_direction_noise_maxs[@]}"; do
            # Another inner loop for c2c_sigmoid_k
            for k_value in "${c2c_sigmoid_k[@]}"; do
                # Another inner loop for c2c_sigmoid_x0
                for x0_value in "${c2c_sigmoid_x0[@]}"; do
                    # Yet another inner loop for c2c_radius_search
                    for radius_value in "${c2c_radius_search[@]}"; do
                        save_name="--save_name rw_speed=${speed}-grad_speed=${grad_speed}-c2c_k=${k_value}-c2c_x0=${x0_value}-c2c_radius=${radius_value}"
                        # Call the second script and pass all arguments
                        sbatch ./scripts/cpuN1C2_param_identification.sbatch "-c" "./examples/vacv-epithelial/dVGFdF11_default_config.ini" $save_name "--randomwalk_speed" "$randomwalk_speed" "--gradient_speed" "$grad_speed" "--gradient_direction_noise_max" "$gradient_direction_noise_max" "--c2c_sigmoid_k" "$k_value" "--c2c_sigmoid_x0" "$x0_value" "--c2c_radius_search" "$radius_value"
                    done
                done
            done
        done
    done
done
