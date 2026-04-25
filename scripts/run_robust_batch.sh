#!/bin/bash

# Configuration ranges for Robust Offline experiments
DATA_TYPES=("robust_syn" "mnist" "adult")
N_VALUES=(1000 5000 10000)
LAYERS_LIST=("16,16" "64,64")
BETAS=(0.01 0.1 1.0)

# Common fixed parameters
RISK_MEASURE="cvar"
ALPHA=0.05
TAU_N=1.0
NUM_SIM=5
NUM_STEPS=1000 # Batch training steps

for data in "${DATA_TYPES[@]}"; do
    for n in "${N_VALUES[@]}"; do
        for layers in "${LAYERS_LIST[@]}"; do
            for beta in "${BETAS[@]}"; do
                echo "=========================================================="
                echo "Running Experiment:"
                echo "Data: $data, N: $n, Layers: $layers, Beta: $beta"
                echo "=========================================================="
                
                python realworld_main.py \
                    --data_type "$data" \
                    --num_contexts "$n" \
                    --layer_sizes "$layers" \
                    --beta "$beta" \
                    --algo_group "robust-offline" \
                    --risk_measure "$RISK_MEASURE" \
                    --alpha "$ALPHA" \
                    --tau_n "$TAU_N" \
                    --num_sim "$NUM_SIM" \
                    --num_steps "$NUM_STEPS" \
                    --nouse_wandb
                
                if [ $? -ne 0 ]; then
                    echo "Experiment failed for $data with n=$n, layers=$layers, beta=$beta"
                    exit 1
                fi
            done
        done
    done
done

echo "All experiments completed successfully!"
