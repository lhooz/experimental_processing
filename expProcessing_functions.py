"""plotting functions for mesh and convergence figures"""

import csv
import os

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, sosfilt, lfilter, freqz


def read_exp_data(expDataFile):
    """read cfd results force coefficients data"""
    expKinematics = []
    expForce = []
    with open(expDataFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count < 1:
                line_count += 1
            else:
                t_motor = row[2]
                # print(row)
                motor_flapping = row[3]
                motor_pitching = row[4]
                expKinematics.append([
                    float(t_motor),
                    float(motor_flapping),
                    float(motor_pitching)
                ])

                t_balance = row[5]
                fx = row[6]
                fy = row[7]
                fz = row[8]
                mx = row[9]
                my = row[10]
                mz = row[11]
                expForce.append([
                    float(t_balance),
                    float(fx),
                    float(fy),
                    float(fz),
                    float(mx),
                    float(my),
                    float(mz),
                ])

                line_count += 1

        print(f'Processed {line_count} lines in {expDataFile}')

    expData = [expKinematics, expForce]

    return expData


def butter_lowpass(cutoff, fs, order=5):
    """lowpass filtering design"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')

    return sos, b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """applying filter to exp data"""
    sos, b, a = butter_lowpass(
        cutoff,
        fs,
        order=order,
    )
    ysos = sosfilt(sos, data)
    yba = lfilter(b, a, data)
    return ysos, yba


def kinematics_processing(data_array,
                          legends,
                          show_range_kinematics,
                          image_out_path,
                          cycle_time,
                          cutoff,
                          b,
                          a,
                          fs,
                          order,
                          plot_frequencyResponse='no'):
    """
    function to plot cfd force coefficients results
    """
    plt.rcParams.update({
        # "text.usetex": True,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'font.size': 24,
        'figure.figsize': (12, 10),
        'lines.linewidth': 4.0,
        'lines.markersize': 0.1,
        'lines.markerfacecolor': 'white',
        'figure.dpi': 300,
        'figure.subplot.left': 0.125,
        'figure.subplot.right': 0.9,
        'figure.subplot.top': 0.9,
        'figure.subplot.bottom': 0.1,
        'figure.subplot.wspace': 0.1,
        'figure.subplot.hspace': 0.1,
    })

    if plot_frequencyResponse == 'yes':
        # Plot the frequency response.
        fig0, axs = plt.subplots(1, 1)
        w, h = freqz(b, a, worN=8000)
        axs.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
        axs.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
        axs.axvline(cutoff, color='k')
        axs.set_xlim(0, 0.5 * fs)
        plt.title("Lowpass Filter Frequency Response")
        axs.set_xlabel('Frequency [Hz]')
        axs.grid()
        # plt.show()

        title = 'refquency respose'
        out_image_file = os.path.join(image_out_path, title + '.svg')
        fig0.savefig(out_image_file)

    legendx = 0.5
    legendy = 1.15

    noOfCases = len(legends)
    range_time = show_range_kinematics[0]
    range_value = show_range_kinematics[1]

    kinematics_arr_processed = []
    fig, axs = plt.subplots(1, 1)
    for i in range(noOfCases):
        kinematics_array = np.array(data_array[i][0])
        timei = (kinematics_array[:, 0] - kinematics_array[0, 0]) / cycle_time
        kinematics_flapping = kinematics_array[:, 1] - kinematics_array[0, 1]
        kinematics_pitching = kinematics_array[:, 2] - kinematics_array[0, 2]
        kinematics_arr_processed.append(
            [timei, kinematics_flapping, kinematics_pitching])

        axs.plot(timei, kinematics_flapping, label=legends[i] + '_flapping')
        axs.plot(timei, kinematics_pitching, label=legends[i] + '_pitching')

    if range_time != 'all':
        axs.set_xlim(range_time)
    if range_value != 'all':
        axs.set_ylim(range_value)

    axs.set_xlabel(r't')
    axs.set_ylabel(r'angle')
    axs.label_outer()

    axs.legend(loc='upper center',
               bbox_to_anchor=(legendx, legendy),
               ncol=len(legends),
               fontsize='small',
               frameon=False)

    title = 'kinematics'
    out_image_file = os.path.join(image_out_path, title + '.svg')
    fig.savefig(out_image_file)
    # plt.show()

    return kinematics_arr_processed


def force_processing(data_array,
                     legends,
                     show_range_kinematics,
                     image_out_path,
                     cycle_time,
                     cutoff,
                     b,
                     a,
                     fs,
                     order,
                     plot_frequencyResponse='no'):
    """
    function to plot cfd force coefficients results
    """
    plt.rcParams.update({
        # "text.usetex": True,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'font.size': 24,
        'figure.figsize': (12, 16),
        'lines.linewidth': 1.0,
        'lines.markersize': 0.1,
        'lines.markerfacecolor': 'white',
        'figure.dpi': 300,
        'figure.subplot.left': 0.125,
        'figure.subplot.right': 0.9,
        'figure.subplot.top': 0.9,
        'figure.subplot.bottom': 0.1,
        'figure.subplot.wspace': 0.1,
        'figure.subplot.hspace': 0.1,
    })

    if plot_frequencyResponse == 'yes':
        # Plot the frequency response.
        fig0, axs = plt.subplots(1, 1)
        w, h = freqz(b, a, worN=8000)
        axs.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
        axs.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
        axs.axvline(cutoff, color='k')
        axs.set_xlim(0, 0.5 * fs)
        plt.title("Lowpass Filter Frequency Response")
        axs.set_xlabel('Frequency [Hz]')
        axs.grid()
        # plt.show()

        title = 'refquency respose'
        out_image_file = os.path.join(image_out_path, title + '.svg')
        fig0.savefig(out_image_file)

    legendx = 0.5
    legendy = 1.15

    noOfCases = len(legends)
    range_time = show_range_kinematics[0]
    range_value = show_range_kinematics[1]

    force_arr_processed = []
    fig, axs = plt.subplots(2, 1)
    for i in range(noOfCases):
        force_array = np.array(data_array[i][1])
        timei = (force_array[:, 0] - force_array[0, 0]) / cycle_time
        fx = force_array[:, 1]
        fy = force_array[:, 2]
        fz = force_array[:, 3]
        mx = force_array[:, 4]
        my = force_array[:, 5]
        mz = force_array[:, 6]
        force_arr_processed.append([timei, fx, fy, fz, mx, my, mz])

        axs[0].plot(timei, fx, label=legends[i] + '_fx')
        axs[0].plot(timei, fy, label=legends[i] + '_fy')
        axs[0].plot(timei, fz, label=legends[i] + '_fz')
        axs[1].plot(timei, mx, label=legends[i] + '_mx')
        axs[1].plot(timei, my, label=legends[i] + '_my')
        axs[1].plot(timei, mz, label=legends[i] + '_mz')

    if range_time != 'all':
        axs[0].set_xlim(range_time)
        axs[1].set_xlim(range_time)
    if range_value != 'all':
        axs[0].set_ylim(range_value)
        axs[1].set_ylim(range_value)

    axs[0].set_ylabel(r'N')
    axs[1].set_ylabel(r'Nmm')
    for ax in axs:
        ax.set_xlabel(r't')
        ax.label_outer()

        ax.legend(loc='upper center',
                  bbox_to_anchor=(legendx, legendy),
                  ncol=len(legends),
                  fontsize='small',
                  frameon=False)

    title = 'force'
    out_image_file = os.path.join(image_out_path, title + '.svg')
    fig.savefig(out_image_file)
    # plt.show()

    return force_arr_processed
