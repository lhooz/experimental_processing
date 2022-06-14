"""script for mesh and convergence figure plots"""

import os
import shutil
import numpy as np
from expPlot_functions import read_processed_data, cf_plotter

# -------------case file control----------
phi = [140]
Re = [3000]
AR = [4]
r1hat = [0.5]
offset = [0.0]
# waveform = ['sinu']
waveform = ['trapzoidal']
# plotMode = 'planform'
plotMode = 'wake'
# figureName = 'Re_effect'
figureName = 'wake_effect_AR3'
#------------------------------------------
# ftc = [0.97]
# ptc = [1.5]
# -----------------------------------------
exp_data_list = []
for phii in phi:
    for r1h in r1hat:
        for ofs in offset:
            for re in Re:
                for ar in AR:
                    exp_data_name = 'phi' + '{0:.1f}'.format(
                        phii) + '__ar' + '{0:.1f}'.format(
                            ar) + '_ofs' + '{0:.1f}'.format(
                                ofs) + '_r1h' + '{0:.1f}'.format(
                                    r1h) + '__Re' + '{0:.1f}'.format(re)
                    exp_data_list.append(exp_data_name)
# -------------time series control----------
time_to_plot = [0, 1.0]
# ------------------------------------------
range_cl = [-0.5, 4]
range_cd = [-2.5, 6.5]
range_cmh = [-1, 5]
range_cmv = [-1.5, 6.5]
show_range = [range_cl, range_cd, range_cmh, range_cmv]
timeScale = 1.0
# ------------------------------------------
cwd = os.getcwd()
data_dir = os.path.join(os.path.dirname(os.path.dirname(cwd)),
                        'processed_results')
out_dir = cwd
# ------------------------------------------
data_array = []
legends = []
for wave in waveform:
    case_dir = os.path.join(data_dir, 'expFinal_processed_' + wave)
    data_folder_all = [f.name for f in os.scandir(case_dir) if f.is_dir()]

    for datai in exp_data_list:
        for data_folder in data_folder_all:
            if data_folder.startswith(datai):
                cfFile = os.path.join(case_dir, data_folder,
                                      'data/aeroForce_coeff.csv')
                cfData = read_processed_data(cfFile)
                data_array.append(cfData)
                legends.append(datai + '_' + wave)

cf_plotter(data_array, legends, time_to_plot, show_range, out_dir, timeScale,
           figureName, plotMode)
