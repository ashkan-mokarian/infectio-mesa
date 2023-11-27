#!/bin/bash

# Define the parameter values
randomwalk_speed=(0.1 0.25 0.5)
gradient_speed=(0.25 0.5 0.75)
c2c_sigmoid_k=(1 2 3)
c2c_sigmoid_x0=(2 3 4)
c2c_radius_search=(4 8 15)

# grid search for parameter identification
for speed in "${randomwalk_speed[@]}"; do
    # Inner loop for gradient_speed
    for grad_speed in "${gradient_speed[@]}"; do
        # Another inner loop for c2c_sigmoid_k
        for k_value in "${c2c_sigmoid_k[@]}"; do
            # Another inner loop for c2c_sigmoid_x0
            for x0_value in "${c2c_sigmoid_x0[@]}"; do
                # Yet another inner loop for c2c_radius_search
                for radius_value in "${c2c_radius_search[@]}"; do
                    save_name="--save_name rw_speed=${speed}-grad_speed=${grad_speed}-c2c_k=${k_value}-c2c_x0=${x0_value}-c2c_radius=${radius_value}"
                    # Call the second script and pass all arguments
                    sbatch ./scripts/cpuN1C2_param_identification.sbatch $save_name "--randomwalk_speed" "$speed" "--gradient_speed" "$grad_speed" "--c2c_sigmoid_k $k_value" "--c2c_sigmoid_x0 $x0_value" "--c2c_radius_search $radius_value"
                done
            done
        done
    done
done
