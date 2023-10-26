#Directory locations and script constants

#COPY & RENAME THIS FILE TO script_variables.py AND EDIT FOR YOUR SETUP

GP_name = 'gp_without_oro_var' # Do NOT include the .pkl file extention

#Different for each user
processed_data_root =   '' #Path to your .npy files from data_prep routine
pngs_root =             '' #Path to where the pngs will be saved to
gp_directory_root =     '' #Path to your gp.pkl files

HYBRID_root =    '' #Path to hybrid directory
HYBRID_data_root = '' #Location .grd files from GP predictions ready for SPEEDY hybrid simulation
SPEEDY_root =    '' #Path to speedy directory

#Different for each dataset
UM_levels = 70
region_count = 80
region_resolution = 448
subregion_count = 4
subregion_resolution = 224

# Flags
TRAIN_GP = False #Change to False to split up workflow and use pre-trained GP
