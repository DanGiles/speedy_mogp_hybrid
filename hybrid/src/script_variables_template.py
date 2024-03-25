#Directory locations and script constants

#COPY & RENAME THIS FILE TO script_variables.py AND EDIT FOR YOUR SETUP

GP_name = 'gp_with_oro_var' # Do NOT include the .pkl file extention

#Different for each user
processed_data_root =   '' #Path to your .npy files from data_prep routine
pngs_root =             '' #Path to where the pngs will be saved to
gp_directory_root =     '' #Path to your gp.pkl files
analysis_root =         '' #Path to analysis .npy files after running SPEEDY

HYBRID_root =    '' #Path to hybrid directory
HYBRID_data_root = '' #Location .grd files from GP predictions ready for SPEEDY hybrid simulation
SPEEDY_root =    '' #Path to speedy directory

#Different for each dataset
UM_levels = 70
region_resolution = 448
subregion_count = 4
region_count = 80
subregion_resolution = 224
num_timesteps = 4
num_days = 10

# Flags
TRAIN_GP = False #Change to False to split up workflow and use pre-trained GP
