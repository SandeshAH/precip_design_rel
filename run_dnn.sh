#!/bin/bash

#########################################################################
# --------------------------------------------------------------------- #
# Author      : Sandesh Athni Hiremath                                  #
# Description : Script to run DNN precipitation model training/testing. #
# Usage       : ./run_dnn.sh [-t | -p | -c]                             #
#   - No arguments : Run training for both GRU and ANN models           #
#   -t | -p        : Run in test mode for GRU and ANN                   #
#   -c             : Run comparison mode for GRU and ANN                #
# --------------------------------------------------------------------- #
#########################################################################


# Navigate to the directory containing the script
cd src
if [ $? -ne 0 ]; then
    echo "Failed to change directory to src"
    exit 1
fi
echo "Changed directory to src"

# Activate python environment
# If the conda environment is not activated, activate it
current_env=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$current_env" != "precip" ]; then
    echo "Current conda environment is not 'precip'."
    echo "Current environment: $current_env"
    # If not, activate it
    source activate precip
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment"
        exit 1
    fi
    echo "Activated conda environment precip"
else
    echo "Current conda environment is 'base'."
fi

if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment"
    exit 1
fi
echo "Activated conda environment precip"

# Check if arguments are passed
if [ $# -eq 1 ]; then
    arg=$1
    if [ "$arg" == "-t" ] || [ "$arg" == "-p" ]; then
        echo "Running in test mode "
        echo "Run GRU testing..."
        python precipitation_dnn_data_fit.py -m test -f best > ../gru_test.out 2>&1 &

        echo "Run ANN testing..."
        python precipitation_dnn_data_fit.py -m test -f best --model_type ann  > ../ann_test.out 2>&1 &
    elif [ "$arg" == "-c" ]; then
        echo "Running comparison mode..."
        python precipitation_dnn_data_fit.py -m cmp -f best --model_type gru  > ../ann_gru_cmp.out 2>&1 &
    fi
    # exit with success code
    exit 0
fi

# Run the Python script
echo "Running the GRU training script..."
(time python precipitation_dnn_data_fit.py -m train --num_epochs 500 > ../gru_train.out 2>&1) >> ../gru_train.out 2>&1 &


echo "Running the ANN training script..."
(time python precipitation_dnn_data_fit.py -m train --model_type ann --num_epochs 500 > ../ann_train.out 2>&1) >> ../ann_train.out 2>&1 &