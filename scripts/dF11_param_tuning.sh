#!/bin/bash

# Define the parameter values
randomwalk_speeds=(0.01 0.1 0.5)
# alphas=(100 200 300) not needed here because VGF particle alphas are known
gradient_speeds=(0.1 0.5 1.0 5)
para_produce_t1s=(0.0 10 25 50)
para_produce_t2s=(30.0 50.0 100.0 150.0)
num_experiments=5

# grid search for parameter identification
for randomwalk_speed in "${randomwalk_speeds[@]}"; do
    for gradient_speed in "${gradient_speeds[@]}"; do
        for para_produce_t1 in "${para_produce_t1s[@]}"; do
            for para_produce_t2 in "${para_produce_t2s[@]}"; do
                # Check if para_produce_t2 is less than para_produce_t1
                if (( $(echo "$para_produce_t2 < $para_produce_t1" | bc -l) )); then
                    continue
                fi
                for ((experiment=1; experiment<=num_experiments; experiment++)); do
                    random_suffix=$((RANDOM % 10000))
                    save_name="randomwalk_speed=${randomwalk_speed}-gradient_speed=${gradient_speed}-para_produce_t1=${para_produce_t1}-para_produce_t2=${para_produce_t2}/${random_suffix}"
                    sbatch ./scripts/cpuN1C2_param_identification.sbatch "-c" "./examples/vacv-epithelial/dF11_default_config.ini" "--save_name" $save_name "--randomwalk_speed" "$randomwalk_speed" "--gradient_speed" "$gradient_speed" "--para_produce_t1" "$para_produce_t1" "--para_produce_t2" "$para_produce_t2"
                done
            done
        done
    done
done