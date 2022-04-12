"""script for mesh and convergence figure plots"""

import os
import shutil
import numpy as np
from expProcessing_functions import read_exp_data, expDataAveraging, butter_lowpass, butter_lowpass_filter
from expProcessing_functions import kinematics_processing, force_processing, force_transform, write_force_array

#-------------case file control----------
phi = [60.0]
Re = [1000.0]
AR = [2.0]
r1hat = [0.4]
offset = [0.0]
ptc = [1.5]
#-----------------------------------------
kinematics_file = 'kinematics.dat'
exp_data_list = []
for phii in phi:
    for r1h in r1hat:
        for ofs in offset:
            for re in Re:
                for ar in AR:
                    for p in ptc:
                        exp_data_name = 'phi' + '{0:.1f}'.format(
                            phii) + '__ar' + '{0:.1f}'.format(
                                ar) + '_ofs' + '{0:.1f}'.format(
                                    ofs) + '_r1h' + '{0:.1f}'.format(
                                        r1h) + '__Re' + '{0:.1f}'.format(
                                            re) + '_ptc' + '{0:.3g}'.format(p)
                        exp_data_list.append(exp_data_name)
# -------------input plot control----------
startTime = 0
endTime = 32
timeStep = 1e-2
outTimeSeries = np.arange(startTime, endTime, timeStep)
# -----------------------------------------
gearRatio = 2.49
# -----------------------------------------
range_time = 'all'
range_value = 'all'
cycle_time = 1.0
# ---------------------------------------
show_range_kinematics = [range_time, range_value]
# -----------------------------------------
cwd = os.getcwd()
data_dir = os.path.join(os.path.dirname(cwd),
                        'experimental_data/expFinal_results')
kinematics_dir = os.path.join(os.path.dirname(cwd),
                              'experimental_data/kinematic_cases_expFinal')
out_dir = os.path.join(os.path.dirname(cwd),
                       'processed_results/expFinal_processed')
# -----------------------------------------
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
#------------------------------------------
data_file_all = [f.name for f in os.scandir(data_dir) if f.is_file()]
air_data_cases = []
water_data_cases = []
for datai in exp_data_list:
    case_out_dir = out_dir + '/' + datai
    image_out_path = case_out_dir + '/images'
    data_out_path = case_out_dir + '/data'

    os.mkdir(case_out_dir)
    os.mkdir(image_out_path)
    os.mkdir(case_out_dir)

    air_data_array = []
    water_data_array = []
    for data_file in data_file_all:
        if data_file.startswith('air_' + datai):
            exp_data_filei = os.path.join(data_dir, data_file)
            zeroDatai, expDatai = read_exp_data(exp_data_filei)
            air_data_array.append([zeroDatai, expDatai])
        elif data_file.startswith('water_' + datai):
            exp_data_filei = os.path.join(data_dir, data_file)
            zeroDatai, expDatai = read_exp_data(exp_data_filei)
            water_data_array.append([zeroDatai, expDatai])

        air_case_average = expDataAveraging(air_data_array)
        water_case_average = expDataAveraging(water_data_array)
        air_data_cases.append(air_case_average)
        water_data_cases.append(water_case_average)
# ---------------------------------------
# Filter requirements.
order = 6
fs = 1 / 64e-3  # sample rate, Hz
cutoff = 20 / 7.5  # desired cutoff frequency of the filter, Hz
# ----------------------------------------------
# Get the filter coefficients so we can check its frequency response.
sos, b, a = butter_lowpass(cutoff, fs, order=order)

kinematics_arr_cases = kinematics_processing(
    exp_data_list, air_data_cases, water_data_cases, gearRatio, outTimeSeries,
    show_range_kinematics, out_dir, cycle_time, cutoff, b, a, fs, order)
force_arr_processed = force_processing(Tweight, data_array, outTimeSeries,
                                       legends, show_range_kinematics,
                                       image_out_path, cycle_time, cutoff, b,
                                       a, fs, order, 'yes')

netForce = force_arr_processed[1] - force_arr_processed[0]

netForce_transformed = force_transform(Tweight, averageKinematics, netForce,
                                       outTimeSeries, show_range_kinematics,
                                       image_out_path)

write_force_array(outTimeSeries, netForce_transformed, data_out_path)
