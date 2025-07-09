#!/bin/bash

#########################################################################
# --------------------------------------------------------------------- #
# Author       : Sandesh Athni Hiremath                                 #
# Description  : Automates training, optimization, plotting, and        #
#                testing of DNN and FBSDE models.                       #
# Usage        : ./runner.sh [-c | -p]                                  #
#   - No args  : Runs training and optimization for all experiments in  #
#                parallel                                               #
#   -c         : Continues optimization from previous checkpoints       #
#   -p         : Only plots results for all experiments                 #
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
    echo "Current conda environment is 'precip'."
fi

if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment"
    exit 1
fi
echo "Activated conda environment precip"

# Check if arguments are passed
if [ $# -eq 1 ]; then
    arg=$1
    if [ "$arg" == "-c" ]; then
        echo "Continuing with optimization..."
        arg="--cont"
    elif [ "$arg" == "-p" ]; then
        echo "Continuing with plotting..."
        python classical_fbsde_adjoint.py -m plot -e 1 >> ../fbsde.out 
        python classical_fbsde_adjoint.py -m plot -e 2 >> ../fbsde.out 
        python classical_fbsde_adjoint.py -m plot -e 3 >> ../fbsde.out 
        python classical_fbsde_adjoint.py -m plot -e 4 >> ../fbsde.out 
        echo "Plotting completed"
        exit 0
    fi
else
    arg=''
fi
# echo "Argument passed: $arg, $1"

echo "Running the GRU training script..."
(time python precipitation_dnn_data_fit.py -m train --num_epochs 500 > ../gru_train.out 2>&1) >> ../gru_train.out 2>&1 &
pid0=$!
echo "Running the ANN training script..."
(time python precipitation_dnn_data_fit.py -m train --model_type ann --num_epochs 500 > ../ann_train.out 2>&1 &) >> ../ann_train.out 2>&1 &
pid1=$!

# Check if the processes started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start DNN processes"
    exit 1
fi
echo "DNN processes started successfully"

# Run the Python script as parallel processes
echo "Running FBSDE algorithm in parallel..."
(time python classical_fbsde_adjoint.py -m opt -N 500 -e 1 $arg > ../fbsde1.out 2>&1 &) >> ../fbsde1.out 2>&1 &
pid2=$!
echo "Started FBSDE process 1 with PID $pid1"
# Run the Python script with different parameters
(time python classical_fbsde_adjoint.py -m opt -N 500 -e 2 $arg >> ../fbsde2.out 2>&1 &) >> ../fbsde2.out 2>&1 &
pid3=$!
echo "Started FBSDE process 2 with PID $pid2"
(time python classical_fbsde_adjoint.py -m opt -N 500 -e 3 $arg >> ../fbsde3.out 2>&1 &) >> ../fbsde3.out 2>&1 &
pid4=$!
echo "Started FBSDE process 3 with PID $pid3"
(time python classical_fbsde_adjoint.py -m opt -N 500 -e 4 $arg >> ../fbsde4.out 2>&1 &) >> ../fbsde4.out 2>&1 &
pid5=$!
echo "Started FBSDE process 4 with PID $pid4"

# Check if the processes started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start FBSDE processes"
    exit 1
fi
echo "FBSDE processes started successfully"

# Wait for all processes to complete
echo "Waiting for all FBSDE processes to complete..."
wait $pid0 $pid1 $pid2 $pid3 $pid4 $pid5
echo "All DNN/FBSDE processes completed."

# Plot the results
echo "Plotting results..."
python classical_fbsde_adjoint.py -m plot -e 1 >> ../fbsde1.out 2>&1 &
pid1=$!
python classical_fbsde_adjoint.py -m plot -e 2 >> ../fbsde2.out 2>&1 &
pid2=$!
python classical_fbsde_adjoint.py -m plot -e 3 >> ../fbsde3.out 2>&1 &
pid3=$!
python classical_fbsde_adjoint.py -m plot -e 4 >> ../fbsde4.out 2>&1 &
pid4=$!

echo "Waiting for plots to be complete..."
wait $pid1 $pid2 $pid3 $pid4

# Check execution status
if [ $? -ne 0 ]; then
    echo "Failed to plot results"
    exit 1
else    
    echo "Plotting results completed successfully"
fi


echo "Run GRU testing..."
python precipitation_dnn_data_fit.py -m test -f best >> ../gru_test.out 2>&1 &
pid1=$!

echo "Run ANN testing..."
python precipitation_dnn_data_fit.py -m test -f best --model_type ann  >> ../ann_test.out 2>&1 &
pid2=$!

echo "Waiting for plots to be complete..."
wait $pid1 $pid2

# Check execution status
if [ $? -ne 0 ]; then
    echo "Failed to plot results"
    exit 1
else    
    echo "Plotting results completed successfully"
fi

echo "Running comparison mode..."
python precipitation_dnn_data_fit.py -m cmp -f best --model_type gru  >> ../ann_gru_cmp.out 

# Check execution status
if [ $? -ne 0 ]; then
    echo "Failed to compare results"
    exit 1
else    
    echo "Result comparison plotted successfully"
fi