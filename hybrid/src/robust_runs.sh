#!/bin/bash
#  nohup bash robust_runs.sh > output.log &
# Range of integers to loop through
start=1
end=5

# Python function to call

# Loop through integers
for ((i=start; i<=end; i++))
do
    # Create directory based on current integer value
    folder="/home/dan/Documents/speedy_mogp_hybrid/results/run_$i"
    export HYBRID_data_root=$folder

    # Call the wrapper
    echo "Calling the wrapper"
    # Start the timer
    start_time=$(date +%s)
    python wrapper.py $HYBRID_data_root
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    # Output time taken
    echo "Python script execution time: $elapsed_time seconds"
    # Postprocess
    echo "Postprocessing"
    python postprocess.py $HYBRID_data_root
done