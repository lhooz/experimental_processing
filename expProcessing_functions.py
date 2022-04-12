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
                    tstart = row[2]
                t_motor = row[2] - tstart
                # print(row)
                motor_flapping = row[3]
                motor_pitching = row[4]
                zeroKinematics.append([
                    float(t_motor),
                    float(motor_flapping),
                    float(motor_pitching)
                ])

                t_balance = row[5] - tstart
                fx = row[6]
                fy = row[7]
                fz = row[8]
                mx = row[9]
                my = row[10]
                mz = row[11]
                zeroForce.append([
                    float(t_balance),
                    float(fx),
                    float(fy),
                    float(fz),
                    float(mx),
                    float(my),
                    float(mz),
                ])

                line_count += 1
            else:
                t_motor = row[2] - tstart
                # print(row)
                motor_flapping = row[3]
                motor_pitching = row[4]
                expKinematics.append([
                    float(t_motor),
                    float(motor_flapping),
                    float(motor_pitching)
                ])

                t_balance = row[5] - tstart
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

    zeroData = [zeroKinematics, zeroForce]
    expData = [expKinematics, expForce]

    return zeroData, expData


def expDataAveraging(data_array):
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
        expForce_average += datai[1][1][0:zero_min_length]

    zeroKinematics_average = zeroKinematics_average / no_of_cases
    expKinematics_average = expKinematics_average / no_of_cases
    zeroForce_average = zeroForce_average / no_of_cases
    expForce_average = expForce_average / no_of_cases

    zero_out = [zeroKinematics_average, zeroForce_average]
    exp_out = [expKinematics_average, expForce_average]
    out_data_array = [zero_out, exp_out]

    return out_data_array


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


def kinematics_processing(exp_data_list,
                          air_data,
                          water_data,
                          gearRatio,
                          outTimeSeries,
                          show_range_kinematics,
                          out_dir,
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
        out_image_file = os.path.join(out_dir, title + '.svg')
        fig0.savefig(out_image_file)

    legendx = 0.5
    legendy = 1.15

    noOfCases = len(exp_data_list)
    range_time = show_range_kinematics[0]
    range_value = show_range_kinematics[1]

    legend = ['air', 'water']
    kinematics_arr_cases = []
    for i in range(noOfCases):
        fig, axs = plt.subplots(1, 1)
        data_array = [air_data[i], water_data[i]]
        kinematics_arr_processed = []
        for j in range(2):
            kinematics_array = np.array(data_array[j][1][0])
            timei = (kinematics_array[:, 0] -
                     kinematics_array[0, 0]) / (1000 * cycle_time)
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

        kinematics_arr_cases.append(averageKinematics)

    kinematics_arr_cases = np.array(kinematics_arr_processed)
    return kinematics_arr_cases


def force_processing(Tweight,
                     data_array,
                     outTimeSeries,
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
        my0 = Tweight[i]
        timei = (force_array[:, 0] - force_array[0, 0]) / (1000 * cycle_time)
        timei = butter_lowpass_filter(timei, cutoff, fs, order)
        fx = butter_lowpass_filter(force_array[:, 1], cutoff, fs, order)
        fy = butter_lowpass_filter(force_array[:, 2], cutoff, fs, order)
        fz = butter_lowpass_filter(force_array[:, 3], cutoff, fs, order)
        mx = butter_lowpass_filter(force_array[:, 4], cutoff, fs, order)
        my = butter_lowpass_filter(force_array[:, 5] - my0, cutoff, fs, order)
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

        axs[0].plot(outTimeSeries, force_arr[:, 0], label=legends[i] + '_fx')
        axs[0].plot(outTimeSeries, force_arr[:, 1], label=legends[i] + '_fy')
        axs[0].plot(outTimeSeries, force_arr[:, 2], label=legends[i] + '_fz')
        axs[1].plot(outTimeSeries, force_arr[:, 3], label=legends[i] + '_mx')
        axs[1].plot(outTimeSeries, force_arr[:, 4], label=legends[i] + '_my')
        axs[1].plot(outTimeSeries, force_arr[:, 5], label=legends[i] + '_mz')

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

    title = 'force'
    out_image_file = os.path.join(image_out_path, title + '.svg')
    fig.savefig(out_image_file)
    # plt.show()

    force_arr_processed = np.array(force_arr_processed)
    return force_arr_processed


def force_transform(Tweight, kinematics_arr, force_arr, outTimeSeries,
                    show_range_kinematics, image_out_path):
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

    mh_Buoyancy = Tweight[0] - Tweight[1]

    legendx = 0.5
    legendy = 1.15

    range_time = show_range_kinematics[0]
    range_value = show_range_kinematics[1]

    fig1, axs = plt.subplots(3, 1)

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

    title = 'net_aeroBuoyant_force_wingFixedFrame'
    out_image_file = os.path.join(image_out_path, title + '.svg')
    fig1.savefig(out_image_file)

    force_arr_tranformed = []
    #-----------------------------------------
    flappingSpline = UnivariateSpline(outTimeSeries, kinematics_arr[:, 0], s=0)
    phiDot = flappingSpline.derivative()
    #-----------------------------------------
    for i in range(len(outTimeSeries)):
        thetai = kinematics_arr[i][1] * np.pi / 180
        fxi = force_arr[i][0]
        fyi = force_arr[i][1]
        fzi = force_arr[i][2]
        mxi = force_arr[i][3]
        myi = force_arr[i][4]
        mzi = force_arr[i][5]

        #--transform to lift-drag frame--
        fhi = np.cos(thetai) * fyi + np.sin(thetai) * fxi
        fvi = -1 * np.sin(thetai) * fyi + np.cos(thetai) * fxi
        fsi = fzi
        mhi = np.cos(thetai) * myi + np.sin(thetai) * mxi - mh_Buoyancy
        mvi = -1 * np.sin(thetai) * myi + np.cos(thetai) * mxi
        msi = mzi

        #---remove direction of mv to resemble drag----
        phiDoti = phiDot(outTimeSeries[i])
        mvi = mvi * np.sign(phiDoti)
        #----------------------------------------------

        force_arr_tranformed.append([fhi, fvi, fsi, mhi, mvi, msi])

    force_arr_tranformed = np.array(force_arr_tranformed)

    fig2, axs2 = plt.subplots(3, 1)

    axs2[0].plot(outTimeSeries, kinematics_arr[:, 0], label='phi')
    axs2[0].plot(outTimeSeries, kinematics_arr[:, 1], label='theta')

    axs2[1].plot(outTimeSeries, force_arr_tranformed[:, 0], label='drag')
    axs2[1].plot(outTimeSeries, force_arr_tranformed[:, 1], label='lift')
    axs2[1].plot(outTimeSeries, force_arr_tranformed[:, 2], label='side_force')
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

    return force_arr_tranformed


def write_force_array(outTimeSeries, force_arr_tranformed, data_out_path):
    """
    function to plot cfd force coefficients results
    """
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
