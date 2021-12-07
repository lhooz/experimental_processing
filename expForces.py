"""script for mesh and convergence figure plots"""

import os
import numpy as np
from expProcessing_functions import read_exp_data, butter_lowpass, butter_lowpass_filter, kinematics_processing, force_processing

#-------------input plot control----------
#-----------------------------------------
exp_data_list = [
    't7_5_air_Run_0_004442',
    't7_5_water_Run_0_023129',
]
#-----------------------------------------
#-----------------------------------------
range_time = 'all'
range_value = 'all'
cycle_time = 1000.0
#---------------------------------------
legends = ['t750_air', 't750_water']
show_range_kinematics = [range_time, range_value]
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

#---------------------------------------
# Filter requirements.
order = 6
fs = 100.0  # sample rate, Hz
cutoff = 10  # desired cutoff frequency of the filter, Hz
#----------------------------------------------
# Get the filter coefficients so we can check its frequency response.
sos, b, a = butter_lowpass(cutoff, fs, order=order)

kinematics_arr_processed = kinematics_processing(data_array, legends,
                                                 show_range_kinematics,
                                                 image_out_path, cycle_time,
                                                 cutoff, b, a, fs, order)
force_arr_processed = force_processing(data_array, legends,
                                       show_range_kinematics, image_out_path,
                                       cycle_time, cutoff, b, a, fs, order,
                                       'yes')
