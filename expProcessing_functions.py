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


def read_motion(kinematics_file):
    """read kinematics file"""
    kinematics_arr = []
    with open(kinematics_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count < 1:
                line_count += 1
            else:
                t = row[0]
                # print(row)
                flapping = row[1]
                pitching = row[2]
                kinematics_arr.append([
                    float(t),
                    float(flapping),
                    float(pitching),
                ])

                line_count += 1

    kinematics_arr = np.array(kinematics_arr)
    return kinematics_arr


def read_refUA(ref_file):
    """read reference velocity and area file"""
    refUA = []
    with open(ref_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                Ur2 = row[1]
                refUA.append(float(Ur2))

                line_count += 1
            elif line_count == 1:
                Ur3 = row[1]
                refUA.append(float(Ur3))

                line_count += 1
            elif line_count == 2:
                r3 = row[1]
                refUA.append(float(r3))

                line_count += 1
            elif line_count == 3:
                Aref = row[1]
                refUA.append(float(Aref))

                line_count += 1

    refUA = np.array(refUA)
    return refUA


def read_exp_data(T, expDataFile):
    """read cfd results force coefficients data"""
    # ---remove final half cycle data---
    initialTime = 20 * 64
    endReadTime = initialTime + 4.5 * T * 1000
    # ----------------------------------

    expKinematics = []
    expForce = []
    zeroKinematics = []
    zeroForce = []
    with open(expDataFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        tstart = 0

        for row in csv_reader:
            if line_count < 1:
                line_count += 1
            elif line_count < 21:
                if line_count == 1:
                    tstart = float(row[2])
                t_motor = float(row[2]) - tstart
                # print(row)
                motor_flapping = row[3]
                motor_pitching = row[4]
                zeroKinematics.append(
                    [t_motor,
                     float(motor_flapping),
                     float(motor_pitching)])

                t_balance = float(row[5]) - tstart
                fx = row[6]
                fy = row[7]
                fz = row[8]
                mx = row[9]
                my = row[10]
                mz = row[11]
                zeroForce.append([
                    t_balance,
                    float(fx),
                    float(fy),
                    float(fz),
                    float(mx),
                    float(my),
                    float(mz),
                ])

                # -------------------------------------
                expKinematics.append(
                    [t_motor,
                     float(motor_flapping),
                     float(motor_pitching)])

                expForce.append([
                    t_balance,
                    float(fx),
                    float(fy),
                    float(fz),
                    float(mx),
                    float(my),
                    float(mz),
                ])
                # -------------------------------------

                line_count += 1
            else:
                t_motor = float(row[2]) - tstart
                # print(row)
                motor_flapping = row[3]
                motor_pitching = row[4]

                t_balance = float(row[5]) - tstart
                fx = row[6]
                fy = row[7]
                fz = row[8]
                mx = row[9]
                my = row[10]
                mz = row[11]

                if t_motor <= endReadTime:
                    expKinematics.append([
                        t_motor,
                        float(motor_flapping),
                        float(motor_pitching)
                    ])

                    expForce.append([
                        t_balance,
                        float(fx),
                        float(fy),
                        float(fz),
                        float(mx),
                        float(my),
                        float(mz),
                    ])

                line_count += 1

        print(f'Processed {line_count} lines in {expDataFile}')

    zeroData = [zeroKinematics, zeroForce]
    expData = [expKinematics, expForce]

    return zeroData, expData


def expDataAveraging(startTime, timeStep, timeScale, data_array):
    """function for averaging multiple exp data"""
    no_of_cases = len(data_array)

    zero_data_length = []
    exp_data_length = []
    for datai in data_array:
        zeroDatai = datai[0][0]
        expDatai = datai[1][0]
        zeroLengthi = len(zeroDatai)
        expLengthi = len(expDatai)
        zero_data_length.append(zeroLengthi)
        exp_data_length.append(expLengthi)

    zero_data_length = np.array(zero_data_length)
    exp_data_length = np.array(exp_data_length)

    zero_min_length = np.min(zero_data_length)
    exp_min_length = np.min(exp_data_length)

    zeroKinematics_average = np.zeros([zero_min_length, 3])
    expKinematics_average = np.zeros([exp_min_length, 3])
    zeroForce_average = np.zeros([zero_min_length, 7])
    expForce_average = np.zeros([exp_min_length, 7])
    for datai in data_array:
        zeroKinematics_average += datai[0][0][0:zero_min_length]
        expKinematics_average += datai[1][0][0:exp_min_length]
        zeroForce_average += datai[0][1][0:zero_min_length]
        expForce_average += datai[1][1][0:exp_min_length]

    zeroKinematics_average = zeroKinematics_average / no_of_cases
    expKinematics_average = expKinematics_average / no_of_cases
    zeroForce_average = zeroForce_average / no_of_cases
    expForce_average = expForce_average / no_of_cases

    # ----time series---
    endTime = expForce_average[-1, 0] / (1000 * timeScale)
    outTimeSeries = np.arange(startTime, endTime, timeStep)
    # -----------------------------------------

    zero_out = [zeroKinematics_average, zeroForce_average]
    exp_out = [expKinematics_average, expForce_average]
    out_data_array = [zero_out, exp_out]

    return outTimeSeries, out_data_array


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
    return ysos


def kinematics_processing(cycleTime_cases,
                          exp_data_list,
                          air_data,
                          water_data,
                          gearRatio,
                          outTimeSeries_all,
                          show_range_kinematics,
                          out_dir,
                          cutoffRatio,
                          timeScale,
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

    legendx = 0.5
    legendy = 1.15

    noOfCases = len(exp_data_list)
    range_time = show_range_kinematics[0]
    range_value = show_range_kinematics[1]

    legends = ['air', 'water']
    filterParameter_cases = []
    kinematics_arr_cases = []
    for i in range(noOfCases):
        fig, axs = plt.subplots(1, 1)
        data_array = [air_data[i], water_data[i]]
        kinematics_arr_processed = []
        outTimeSeries = outTimeSeries_all[i]

        for j in range(2):
            kinematics_array = np.array(data_array[j][1][0])
            timei = (kinematics_array[:, 0] -
                     kinematics_array[0, 0]) / (1000 * timeScale)
            # print(timei)
            kinematics_flapping = kinematics_array[:, 1] - kinematics_array[0,
                                                                            1]
            kinematics_pitching = kinematics_array[:, 2] - kinematics_array[0,
                                                                            2]

            flappingSpline = UnivariateSpline(timei, kinematics_flapping, s=0)
            pitchingSpline = UnivariateSpline(timei, kinematics_pitching, s=0)

            kinematics_arr = []
            for ti in outTimeSeries:
                phii = flappingSpline(ti) / 4096 * 360
                thetai = (pitchingSpline(ti) -
                          flappingSpline(ti) / gearRatio) / 4096 * 360
                # thetai = pitchingSpline(ti) / 4096 * 360
                kinematic_anglei = [phii, thetai]
                kinematics_arr.append(kinematic_anglei)

            kinematics_arr = np.array(kinematics_arr)
            kinematics_arr_processed.append(kinematics_arr)

            axs.plot(outTimeSeries,
                     kinematics_arr[:, 0],
                     label=legends[j] + '_flapping')
            axs.plot(outTimeSeries,
                     kinematics_arr[:, 1],
                     label=legends[j] + '_pitching')

        averageKinematics = 0.5 * (kinematics_arr_processed[0] +
                                   kinematics_arr_processed[1])

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
        case_out_dir = out_dir + '/' + exp_data_list[i]
        image_out_path = case_out_dir + '/images'
        out_image_file = os.path.join(image_out_path, title + '.svg')
        fig.savefig(out_image_file)
        # plt.show()

        kinematics_arr_cases.append(np.array(averageKinematics))

        # desired cutoff frequency of the filter, Hz
        cutoff = cutoffRatio / cycleTime_cases[i]
        # Get the filter coefficients so we can check its frequency response.
        sos, b, a = butter_lowpass(cutoff, fs, order=order)
        filterParameter_cases.append([cutoff, b, a])
        # ---------------------------------------------------------

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

        plt.close('all')
        # plt.show()

    return filterParameter_cases, kinematics_arr_cases


def force_processing(exp_data_list,
                     air_data,
                     water_data,
                     outTimeSeries_all,
                     show_range_kinematics,
                     out_dir,
                     timeScale,
                     filterParameter_cases,
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

    legendx = 0.5
    legendy = 1.15

    noOfCases = len(exp_data_list)
    range_time = show_range_kinematics[0]
    range_value = show_range_kinematics[1]

    legends = ['air', 'water']
    bouyacyF_cases = []
    netForce_cases = []
    for j in range(noOfCases):
        fig, axs = plt.subplots(2, 1)
        cutoff = filterParameter_cases[j][0]
        zero_array = [air_data[j][0][1], water_data[j][0][1]]
        data_array = [air_data[j][1], water_data[j][1]]
        force_arr_processed = []
        outTimeSeries = outTimeSeries_all[j]

        for i in range(2):
            force_array = np.array(data_array[i][1])
            timei = force_array[:, 0] / (1000 * timeScale)
            timei = butter_lowpass_filter(timei, cutoff, fs, order)
            fx = butter_lowpass_filter(force_array[:, 1], cutoff, fs, order)
            fy = butter_lowpass_filter(force_array[:, 2], cutoff, fs, order)
            fz = butter_lowpass_filter(force_array[:, 3], cutoff, fs, order)
            mx = butter_lowpass_filter(force_array[:, 4], cutoff, fs, order)
            my = butter_lowpass_filter(force_array[:, 5], cutoff, fs, order)
            mz = butter_lowpass_filter(force_array[:, 6], cutoff, fs, order)

            fxSpline = UnivariateSpline(timei, fx, s=0)
            fySpline = UnivariateSpline(timei, fy, s=0)
            fzSpline = UnivariateSpline(timei, fz, s=0)
            mxSpline = UnivariateSpline(timei, mx, s=0)
            mySpline = UnivariateSpline(timei, my, s=0)
            mzSpline = UnivariateSpline(timei, mz, s=0)

            force_arr = []
            for ti in outTimeSeries:
                forcei = [
                    fxSpline(ti),
                    fySpline(ti),
                    fzSpline(ti),
                    mxSpline(ti),
                    mySpline(ti),
                    mzSpline(ti)
                ]
                # print(forcei[0])
                force_arr.append(forcei)

            force_arr = np.array(force_arr)
            force_arr_processed.append(force_arr)

            axs[0].plot(outTimeSeries,
                        force_arr[:, 0],
                        label=legends[i] + '_fx')
            axs[0].plot(outTimeSeries,
                        force_arr[:, 1],
                        label=legends[i] + '_fy')
            axs[0].plot(outTimeSeries,
                        force_arr[:, 2],
                        label=legends[i] + '_fz')
            axs[1].plot(outTimeSeries,
                        force_arr[:, 3],
                        label=legends[i] + '_mx')
            axs[1].plot(outTimeSeries,
                        force_arr[:, 4],
                        label=legends[i] + '_my')
            axs[1].plot(outTimeSeries,
                        force_arr[:, 5],
                        label=legends[i] + '_mz')

            # axs[0].plot(timei, fx, label=legends[i] + '_fx')
            # axs[0].plot(timei, fy, label=legends[i] + '_fy')
            # axs[0].plot(timei, fz, label=legends[i] + '_fz')
            # axs[1].plot(timei, mx, label=legends[i] + '_mx')
            # axs[1].plot(timei, my, label=legends[i] + '_my')
            # axs[1].plot(timei, mz, label=legends[i] + '_mz')

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

        title = 'balance forces'
        case_out_dir = out_dir + '/' + exp_data_list[j]
        image_out_path = case_out_dir + '/images'
        out_image_file = os.path.join(image_out_path, title + '.svg')
        fig.savefig(out_image_file)
        # plt.show()

        # ----Bouyacy average----
        netZero = []
        for i in range(2):
            no_samples = len(zero_array[i])
            averageZero = np.zeros(7)
            for datai in zero_array[i]:
                averageZero += datai

            averageZero = averageZero / no_samples
            netZero.append(averageZero)
        bouyacyF = np.array(netZero[1]) - np.array(netZero[0])
        bouyacyF_cases.append(bouyacyF)

        # ----net aerodynamic force-----
        netForce = force_arr_processed[1] - force_arr_processed[0]
        netForce_cases.append(netForce)

        # ----plotting filter frequency response----
        b = filterParameter_cases[j][1]
        a = filterParameter_cases[j][2]
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

            title = 'refquency respose'
            out_image_file = os.path.join(image_out_path, title + '.svg')
            fig0.savefig(out_image_file)

        plt.close('all')
        # plt.show()

    return bouyacyF_cases, netForce_cases


def force_transform(exp_data_list, kinematics_arr_cases, bouyacy_arr_cases,
                    force_arr_cases, outTimeSeries_all, show_range_kinematics,
                    out_dir):
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

    legendx = 0.5
    legendy = 1.15

    noOfCases = len(exp_data_list)

    range_time = show_range_kinematics[0]
    range_value = show_range_kinematics[1]

    transformed_aeroForce_cases = []
    for i in range(noOfCases):
        outTimeSeries = outTimeSeries_all[i]
        fig1, axs = plt.subplots(3, 1)

        kinematics_arr = kinematics_arr_cases[i]
        force_arr = force_arr_cases[i]
        b = bouyacy_arr_cases[i]
        bouyacy_ldFrame = [b[2], b[1], b[3], b[5], b[4], b[6]]

        axs[0].plot(outTimeSeries, kinematics_arr[:, 0], label='phi')
        axs[0].plot(outTimeSeries, kinematics_arr[:, 1], label='theta')

        axs[1].plot(outTimeSeries, force_arr[:, 0], label='net_fx')
        axs[1].plot(outTimeSeries, force_arr[:, 1], label='net_fy')
        axs[1].plot(outTimeSeries, force_arr[:, 2], label='net_fz')
        axs[2].plot(outTimeSeries, force_arr[:, 3], label='net_mx')
        axs[2].plot(outTimeSeries, force_arr[:, 4], label='net_my')
        axs[2].plot(outTimeSeries, force_arr[:, 5], label='net_mz')

        axs[0].set_ylabel(r'Angle (degrees)')
        axs[1].set_ylabel(r'Force (N)')
        axs[2].set_ylabel(r'Moment (Nmm)')
        for ax in axs:
            ax.set_xlabel(r't')
            ax.label_outer()

            ax.legend(loc='upper center',
                      bbox_to_anchor=(legendx, legendy),
                      ncol=3,
                      fontsize='small',
                      frameon=False)

        title = 'net_aero_force_wingFixedFrame_withBouyacy'
        case_out_dir = out_dir + '/' + exp_data_list[i]
        image_out_path = case_out_dir + '/images'
        out_image_file = os.path.join(image_out_path, title + '.svg')
        fig1.savefig(out_image_file)

        force_arr_tranformed = []
        # -----------------------------------------
        flappingSpline = UnivariateSpline(outTimeSeries,
                                          kinematics_arr[:, 0],
                                          s=0)
        phiDot = flappingSpline.derivative()
        # -----------------------------------------
        for i in range(len(outTimeSeries)):
            thetai = kinematics_arr[i][1] * np.pi / 180
            fxi = force_arr[i][0]
            fyi = force_arr[i][1]
            fzi = force_arr[i][2]
            mxi = force_arr[i][3]
            myi = force_arr[i][4]
            mzi = force_arr[i][5]

            # --transform to lift-drag frame--
            fhi = np.cos(thetai) * fyi + np.sin(
                thetai) * fxi - bouyacy_ldFrame[0]
            fvi = -1 * np.sin(thetai) * fyi + np.cos(
                thetai) * fxi - bouyacy_ldFrame[1]
            fsi = fzi - bouyacy_ldFrame[2]
            mhi = np.cos(thetai) * myi + np.sin(
                thetai) * mxi - bouyacy_ldFrame[3]
            mvi = -1 * np.sin(thetai) * myi + np.cos(
                thetai) * mxi - bouyacy_ldFrame[4]
            msi = mzi - bouyacy_ldFrame[5]

            # ---remove direction of fh, mv to resemble drag----
            phiDoti = phiDot(outTimeSeries[i])
            fhi = -1 * fhi * np.sign(phiDoti)
            mvi = mvi * np.sign(phiDoti)
            # ----------------------------------------------

            force_arr_tranformed.append([fhi, fvi, fsi, mhi, mvi, msi])

        force_arr_tranformed = np.array(force_arr_tranformed)

        fig2, axs2 = plt.subplots(3, 1)

        axs2[0].plot(outTimeSeries, kinematics_arr[:, 0], label='phi')
        axs2[0].plot(outTimeSeries, kinematics_arr[:, 1], label='theta')

        axs2[1].plot(outTimeSeries, force_arr_tranformed[:, 0], label='drag')
        axs2[1].plot(outTimeSeries, force_arr_tranformed[:, 1], label='lift')
        axs2[1].plot(outTimeSeries,
                     force_arr_tranformed[:, 2],
                     label='side_force')
        axs2[2].plot(outTimeSeries, force_arr_tranformed[:, 3], label='mh')
        axs2[2].plot(outTimeSeries, force_arr_tranformed[:, 4], label='mv')
        axs2[2].plot(outTimeSeries, force_arr_tranformed[:, 5], label='ms')

        axs2[0].set_ylabel(r'Angle (degrees)')
        axs2[1].set_ylabel(r'Force (N)')
        axs2[2].set_ylabel(r'Moment (Nmm)')
        for ax in axs2:
            ax.set_xlabel(r't')
            ax.label_outer()

            ax.legend(loc='upper center',
                      bbox_to_anchor=(legendx, legendy),
                      ncol=3,
                      fontsize='small',
                      frameon=False)

        title = 'net_aerodForce_transformed'
        out_image_file = os.path.join(image_out_path, title + '.svg')
        fig2.savefig(out_image_file)

        transformed_aeroForce_cases.append(force_arr_tranformed)

        plt.close('all')
        # plt.show()

    return transformed_aeroForce_cases


def write_force_array(exp_data_list, outTimeSeries_all,
                      transformed_aeroForce_cases, out_dir):
    """
    function to write exp force results
    """
    noOfCases = len(exp_data_list)

    for i in range(noOfCases):
        outTimeSeries = outTimeSeries_all[i]
        force_arr_tranformed = transformed_aeroForce_cases[i]
        case_out_dir = out_dir + '/' + exp_data_list[i]
        data_out_path = case_out_dir + '/data'

        data = []
        for ti, fm_i in zip(outTimeSeries, force_arr_tranformed):
            fm_str = []
            for fm in fm_i:
                fm_s = '{0:.10g}'.format(fm)
                fm_str.append(fm_s)

            datai = '{0:.10g}'.format(ti) + ',' + ','.join(fm_str)
            data.append(datai)

        save_file = data_out_path + '/net_aeroForce_transformed.csv'

        with open(save_file, 'w') as f:
            f.write("t(s),fx(N),fy(N),fz(N),mh(Nmm),mv(Nmm),ms(Nmm)\n")
            for item in data:
                f.write("%s\n" % item)


def write_coeff_array(coeff_ref_cases, cycleTime_cases, exp_data_list,
                      outTimeSeries_all, transformed_aeroForce_cases, out_dir,
                      medium):
    """
    function to write exp coefficients results
    """
    initialTime = 20 * 64 / 1000

    noOfCases = len(exp_data_list)

    if medium == 'water':
        rho = 999.7
        nu = 1.3065e-6
    elif medium == 'air':
        rho = 1.246
        nu = 1.426e-5

    for i in range(noOfCases):
        outTimeSeries = outTimeSeries_all[i]
        force_arr_tranformed = transformed_aeroForce_cases[i]
        coeff_data = coeff_ref_cases[i]
        T = cycleTime_cases[i]
        case_out_dir = out_dir + '/' + exp_data_list[i]
        data_out_path = case_out_dir + '/data'

        f_scale = 0.5 * rho * coeff_data[0]**2 * coeff_data[3]
        m_scale = 0.5 * rho * coeff_data[1]**2 * coeff_data[2] * coeff_data[
            3] * 1000

        data = []
        for ti, fm_i in zip(outTimeSeries, force_arr_tranformed):
            t_hat = (ti - initialTime) / T

            cfm_i = [
                fm_i[0] / f_scale, fm_i[1] / f_scale, fm_i[2] / f_scale,
                fm_i[3] / m_scale, fm_i[4] / m_scale, fm_i[5] / m_scale
            ]
            fm_str = []
            for fm in cfm_i:
                fm_s = '{0:.10g}'.format(fm)
                fm_str.append(fm_s)

            datai = '{0:.10g}'.format(t_hat) + ',' + ','.join(fm_str)
            data.append(datai)

        save_file = data_out_path + '/aeroForce_coeff.csv'

        with open(save_file, 'w') as f:
            f.write("t_hat,cfx,cfy,cfz,cmh,cmv,cms\n")
            for item in data:
                f.write("%s\n" % item)
