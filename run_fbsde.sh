#!/bin/bash

#########################################################################
# --------------------------------------------------------------------- #
# Author      : Sandesh Athni Hiremath                                  #
# Description : Script to run FBSDE optimization and plotting.          #
# Usage: ./run_fbsde.sh [-c | -p]                                       #
#   - No arguments : Run optimization for all experiments in parallel   #
#   -c            : Continue optimization from previous state           #
#   -p            : Plot results for all experiments                    #
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
if [ "$current_env" != "base" ]; then
    echo "Current conda environment is not 'base'."
    echo "Current environment: $current_env"
    # If not, activate it
    source activate base
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment"
        exit 1
    fi
    echo "Activated conda environment base"
else
    echo "Current conda environment is 'base'."
fi

if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment"
    exit 1
fi
echo "Activated conda environment base"

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

# Run the Python script as parallel processes
echo "Running FBSDE algorithm in parallel..."
python classical_fbsde_adjoint.py -m opt -N 500 -e 1 $arg > ../fbsde.out 2>&1 &
pid1=$!
echo "Started FBSDE process 1 with PID $pid1"
# Run the Python script with different parameters
python classical_fbsde_adjoint.py -m opt -N 500 -e 2 $arg >> ../fbsde.out 2>&1 &
pid2=$!
echo "Started FBSDE process 2 with PID $pid2"
python classical_fbsde_adjoint.py -m opt -N 500 -e 3 $arg >> ../fbsde.out 2>&1 &
pid3=$!
echo "Started FBSDE process 3 with PID $pid3"
python classical_fbsde_adjoint.py -m opt -N 500 -e 4 $arg >> ../fbsde.out 2>&1 &
pid4=$!
echo "Started FBSDE process 4 with PID $pid4"

# Check if the processes started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start FBSDE processes"
    exit 1
fi
echo "FBSDE processes started successfully"

# Wait for all processes to complete
echo "Waiting for all FBSDE processes to complete..."
wait $pid1 $pid2 $pid3 $pid4
echo "All FBSDE processes completed."

# Plot the results
echo "Plotting results..."
python classical_fbsde_adjoint.py -m plot -e 1 >> ../fbsde.out 2>&1 &
pid1=$!
python classical_fbsde_adjoint.py -m plot -e 2 >> ../fbsde.out 2>&1 &
pid2=$!
python classical_fbsde_adjoint.py -m plot -e 3 >> ../fbsde.out 2>&1 &
pid3=$!
python classical_fbsde_adjoint.py -m plot -e 4 >> ../fbsde.out 2>&1 &
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