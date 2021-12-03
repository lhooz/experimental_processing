"""script for mesh and convergence figure plots"""

import os

from expProcessing_functions import read_exp_data

#-------------input plot control----------
#-----------------------------------------
exp_data_list = [
    't7_5_air_Run_0_004442',
    't7_5_water_Run_0_023129',
]
#-----------------------------------------
#-----------------------------------------
time_to_plot = 'all'
coeffs_show_range = 'all'
time_to_plot = [4.0, 5.0]
show_range_cl = [-0.2, 2.5]
show_range_cd = [-1.2, 2.8]
cycle_time = 1.0
#---------------------------------------
show_range = [show_range_cl, show_range_cd]
#-----------------------------------------
cwd = os.getcwd()
data_dir = os.path.join(os.path.dirname(cwd), 'test_data')
image_out_path = os.path.join(os.path.dirname(cwd), 'processed_results')
#-----------------------------------------
data_file_all = [f.name for f in os.scandir(data_dir) if f.is_file()]
data_array = []
for datai in exp_data_list:
    for data_file in data_file_all:
        if data_file.startswith(datai):
            exp_data_filei = os.path.join(data_dir, data_file)
            expDatai = read_exp_data(exp_data_filei)
            data_array.append(expDatai)

print(data_array)
#---------------------------------------
