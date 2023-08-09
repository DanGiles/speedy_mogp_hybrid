#!/bin/bash -l

# Batch script to run a serial job under SGE.

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:00

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=5G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=20G

# Set the name of the job.
#$ -N speedy-hybrid

# Request 32 cores.
#$ -pe smp 36

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/ucakjcb/Scratch/clouds

# Your work should be done in $TMPDIR 
cd /home/ucakjcb/ml_climate_fusion/hybrid/src/

# activate a virtual python environment with mogp-emulator version 0.6.1
module load python3/recommended
source /home/ucakjcb/venvs/mogp061/bin/activate
# source /home/ucakdpg/Scratch/mogp-speedy/mogp/bin/activate

# Run the application and put the output into a file called date.txt
python /home/ucakjcb/ml_climate_fusion/hybrid/src/wrapper.py