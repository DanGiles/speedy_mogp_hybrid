#Directory locations and script constants

#COPY & RENAME THIS FILE TO script_variables.py AND EDIT FOR YOUR SETUP

#Different for each user
processed_data_root = '' #This points to where your .npy files are
pngs_root = '' #This is where the pngs will be saved to
gp_directory = '' #This points to where your gp.pkl files will go

#Different for each dataset
UM_levels = 70
region_count = 80 #was 99 (but actually about 94)
region_resolution = 448 #was 196?
subregion_count = 4
subregion_resolution = 224