#!/bin/bash -l

# Batch script to run a serial job under SGE.

# Request 12 hours of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=4:00:00

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=10G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=1G

# Set the name of the job.
#$ -N calc-field-avg

# Request 4 cores.
#$ -pe smp 4

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/ucakjcb/Scratch/clouds

# Your work should be done in $TMPDIR 
cd /home/ucakjcb/ml_climate_fusion/speedy_fusion/src/

module load python3/recommended

# Run the application and put the output into a file called date.txt
python /home/ucakjcb/ml_climate_fusion/analysis/create_mean_fields.py
