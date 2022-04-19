"""script for mesh and convergence figure plots"""

import os
import shutil
import numpy as np
from expProcessing_functions import read_exp_data, expDataAveraging, butter_lowpass, butter_lowpass_filter
from expProcessing_functions import kinematics_processing, force_processing, force_transform, write_force_array
from expProcessing_functions import read_motion, read_refUA, write_coeff_array

# -------------case file control----------
phi = [90, 140.0]
Re = [5000.0]
AR = [5.0]
r1hat = [0.4, 0.5, 0.6]
offset = [0.0]
ptc = [1.5]
#------------- Filter requirements-----------
order = 6
fs = 1 / 64e-3  # sample rate, Hz
cutoffRatio = 10  # how many times of flapping frequency to be cutoff
# -----------------------------------------
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
# -------------time series control----------
startTime = 0
timeStep = 1e-2
# -----------------------------------------
gearRatio = 2.49
# -----------------------------------------
range_time = 'all'
range_value = 'all'
timeScale = 1.0
# ---------------------------------------
show_range_kinematics = [range_time, range_value]
# -----------------------------------------
cwd = os.getcwd()
data_dir = os.path.join(os.path.dirname(cwd),
                        'experimental_data/expFinal_results')
refUA_dir = os.path.join(os.path.dirname(cwd),
                         'experimental_data/kinematic_cases_expFinal')
out_dir = os.path.join(os.path.dirname(cwd),
                       'processed_results/expFinal_processed')
# -----------------------------------------
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
# ------------------------------------------
data_file_all = [f.name for f in os.scandir(data_dir) if f.is_file()]
ref_file_all = [f.name for f in os.scandir(refUA_dir) if f.is_file()]
cycleTime_cases = []
coeff_ref_cases = []
outTimeSeries = []
air_data_cases = []
water_data_cases = []
for datai in exp_data_list:
    case_out_dir = out_dir + '/' + datai
    image_out_path = case_out_dir + '/images'
    data_out_path = case_out_dir + '/data'

    if os.path.exists(case_out_dir):
        shutil.rmtree(case_out_dir)
    os.mkdir(case_out_dir)
    os.mkdir(image_out_path)
    os.mkdir(data_out_path)

    T = 0
    for refFile in ref_file_all:
        if refFile.startswith(datai) and refFile.endswith('.csv'):
            kinematics_file = os.path.join(refUA_dir, refFile)
            inKinematics = read_motion(kinematics_file)
            T = inKinematics[-1, 0]
            cycleTime_cases.append(T)
        if refFile.startswith(datai) and refFile.endswith('.cf'):
            cf_file = os.path.join(refUA_dir, refFile)
            ref_data = read_refUA(cf_file)
            coeff_ref_cases.append(ref_data)

    air_data_array = []
    water_data_array = []
    for data_file in data_file_all:
        if data_file.startswith('air_' + datai):
            exp_data_filei = os.path.join(data_dir, data_file)
            zeroDatai, expDatai = read_exp_data(T, exp_data_filei)
            air_data_array.append([zeroDatai, expDatai])
        elif data_file.startswith('water_' + datai):
            exp_data_filei = os.path.join(data_dir, data_file)
            zeroDatai, expDatai = read_exp_data(T, exp_data_filei)
            water_data_array.append([zeroDatai, expDatai])

    air_time, air_case_average = expDataAveraging(startTime, timeStep,
                                                  timeScale, air_data_array)
    water_time, water_case_average = expDataAveraging(startTime, timeStep,
                                                      timeScale,
                                                      water_data_array)
    if len(air_time) <= len(water_time):
        case_time_series = air_time
    else:
        case_time_series = water_time

    outTimeSeries.append(case_time_series)
    air_data_cases.append(air_case_average)
    water_data_cases.append(water_case_average)
# ---------------------------------------
filterParameter_cases, kinematics_arr_cases = kinematics_processing(
    cycleTime_cases, exp_data_list, air_data_cases, water_data_cases,
    gearRatio, outTimeSeries, show_range_kinematics, out_dir, cutoffRatio,
    timeScale, fs, order)
bouyacyF_cases, netForce_cases = force_processing(
    exp_data_list, air_data_cases, water_data_cases, outTimeSeries,
    show_range_kinematics, out_dir, timeScale, filterParameter_cases, fs,
    order, 'yes')

transformed_aeroForce_cases = force_transform(exp_data_list,
                                              kinematics_arr_cases,
                                              bouyacyF_cases, netForce_cases,
                                              outTimeSeries,
                                              show_range_kinematics, out_dir)

write_force_array(exp_data_list, outTimeSeries, transformed_aeroForce_cases,
                  out_dir)
write_coeff_array(coeff_ref_cases, cycleTime_cases, exp_data_list,
                  outTimeSeries, transformed_aeroForce_cases, out_dir, 'water')
