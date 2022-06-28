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


def read_processed_data(expDataFile):
    """read cfd results force coefficients data"""
    # ---remove final half cycle data---

    expForceCoeff = []
    with open(expDataFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        tstart = 0

        for row in csv_reader:
            if line_count < 1:
                line_count += 1
            else:
                t_hat = row[0]
                cd = row[1]
                cl = row[2]
                cs = row[3]
                cmh = row[4]
                cmv = row[5]
                cms = row[6]
                phi = row[7]
                theta = row[8]
                expForceCoeff.append([
                    float(t_hat),
                    float(cd),
                    float(cl),
                    float(cs),
                    float(cmh),
                    float(cmv),
                    float(cms),
                    float(phi),
                    float(theta),
                ])
                # -------------------------------------

                line_count += 1
        print(f'Processed {line_count} lines in {expDataFile}')
        expForceCoeff = np.array(expForceCoeff)

    return expForceCoeff


def cf_plotter(data_array, legends, time_to_plot, show_range, out_dir,
               timeScale, figureName, plotMode):
    """
    function to plot cfd force coefficients results
    """
    plt.rcParams.update({
        # "text.usetex": True,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'font.size': 20,
        'figure.figsize': (15, 30),
        # 'figure.figsize': (40, 30),
        'lines.linewidth': 2.0,
        'lines.markersize': 0.1,
        'lines.markerfacecolor': 'white',
        'figure.dpi': 300,
        'figure.subplot.left': 0.35,
        'figure.subplot.right': 0.65,
        'figure.subplot.top': 0.7,
        'figure.subplot.bottom': 0.1,
        'figure.subplot.wspace': 0.2,
        'figure.subplot.hspace': 0.25,
    })
    plot_cycle = 4
    legendx = 0.4
    legendy = 3.2

    cf_array = data_array
    range_cl = show_range[0]
    range_cd = show_range[1]
    range_cmh = show_range[2]
    range_cmv = show_range[3]

    timeSeries = np.linspace(time_to_plot[0], time_to_plot[1], 1000)
    fig, axs = plt.subplots(5, 1)
    mcl_1st_arr = []
    mcl_4th_arr = []
    mcl_wake_arr = []
    mcl_wake_mid_arr = []

    mcd_1st_arr = []
    mcd_4th_arr = []
    mcd_wake_arr = []
    mcd_wake_mid_arr = []

    mcmh_1st_arr = []
    mcmh_4th_arr = []
    mcmh_wake_arr = []
    mcmh_wake_mid_arr = []

    mcmv_1st_arr = []
    mcmv_4th_arr = []
    mcmv_wake_arr = []
    mcmv_wake_mid_arr = []

    for i in range(len(legends)):
        phi_spl = UnivariateSpline(cf_array[i][:, 0], cf_array[i][:, 7], s=0)
        theta_spl = UnivariateSpline(cf_array[i][:, 0], cf_array[i][:, 8], s=0)

        cl_spl_1st = UnivariateSpline(cf_array[i][:, 0],
                                      cf_array[i][:, 2],
                                      s=0)
        cl_spl_4th = UnivariateSpline(cf_array[i][:, 0] - plot_cycle + 1,
                                      cf_array[i][:, 2],
                                      s=0)

        cd_spl_1st = UnivariateSpline(cf_array[i][:, 0],
                                      cf_array[i][:, 1],
                                      s=0)
        cd_spl_4th = UnivariateSpline(cf_array[i][:, 0] - plot_cycle + 1,
                                      cf_array[i][:, 1],
                                      s=0)

        cmh_spl_1st = UnivariateSpline(cf_array[i][:, 0],
                                       cf_array[i][:, 4],
                                       s=0)
        cmh_spl_4th = UnivariateSpline(cf_array[i][:, 0] - plot_cycle + 1,
                                       cf_array[i][:, 4],
                                       s=0)

        cmv_spl_1st = UnivariateSpline(cf_array[i][:, 0],
                                       cf_array[i][:, 5],
                                       s=0)
        cmv_spl_4th = UnivariateSpline(cf_array[i][:, 0] - plot_cycle + 1,
                                       cf_array[i][:, 5],
                                       s=0)

        phi = []
        theta = []

        cl_1st = []
        cl_4th = []
        cl_wake = []

        cd_1st = []
        cd_4th = []
        cd_wake = []

        cmh_1st = []
        cmh_4th = []
        cmh_wake = []

        cmv_1st = []
        cmv_4th = []
        cmv_wake = []

        for ti in timeSeries:
            phi.append(phi_spl(ti))
            theta.append(theta_spl(ti))

            cl_1st.append(cl_spl_1st(ti))
            cl_4th.append(cl_spl_4th(ti))
            cl_wake.append(cl_spl_4th(ti) - cl_spl_1st(ti))

            cd_1st.append(cd_spl_1st(ti))
            cd_4th.append(cd_spl_4th(ti))
            cd_wake.append(cd_spl_4th(ti) - cd_spl_1st(ti))

            cmh_1st.append(cmh_spl_1st(ti))
            cmh_4th.append(cmh_spl_4th(ti))
            cmh_wake.append(cmh_spl_4th(ti) - cmh_spl_1st(ti))

            cmv_1st.append(cmv_spl_1st(ti))
            cmv_4th.append(cmv_spl_4th(ti))
            cmv_wake.append(cmv_spl_4th(ti) - cmv_spl_1st(ti))

        if plotMode == 'wake':
            axs[1].plot(timeSeries / timeScale,
                        cl_1st,
                        label=legends[i] + '_1st')
            axs[1].plot(timeSeries / timeScale,
                        cl_wake,
                        label=legends[i] + '_wake')

            axs[2].plot(timeSeries / timeScale,
                        cd_1st,
                        label=legends[i] + '_1st')
            axs[2].plot(timeSeries / timeScale,
                        cd_wake,
                        label=legends[i] + '_wake')

            axs[3].plot(timeSeries / timeScale,
                        cmh_1st,
                        label=legends[i] + '_1st')
            axs[3].plot(timeSeries / timeScale,
                        cl_wake,
                        label=legends[i] + '_wake')

            axs[4].plot(timeSeries / timeScale,
                        cmv_1st,
                        label=legends[i] + '_1st')
            axs[4].plot(timeSeries / timeScale,
                        cmv_wake,
                        label=legends[i] + '_wake')

        axs[0].plot(timeSeries / timeScale, phi, label=legends[i] + '_phi')
        axs[0].plot(timeSeries / timeScale, theta, label=legends[i] + '_theta')
        axs[1].plot(timeSeries / timeScale,
                    cl_4th,
                    label=legends[i] + '_' + '{0:0g}'.format(plot_cycle) +
                    'th')
        axs[2].plot(timeSeries / timeScale,
                    cd_4th,
                    label=legends[i] + '_' + '{0:0g}'.format(plot_cycle) +
                    'th')
        axs[3].plot(timeSeries / timeScale,
                    cmh_4th,
                    label=legends[i] + '_' + '{0:0g}'.format(plot_cycle) +
                    'th')
        axs[4].plot(timeSeries / timeScale,
                    cmv_4th,
                    label=legends[i] + '_' + '{0:0g}'.format(plot_cycle) +
                    'th')
        #------------half stroke averages-----------
        mcl_1st = cl_spl_1st.integral(0, 0.5)
        mcl_4th = cl_spl_4th.integral(0, 0.5)
        mcl_wake = mcl_4th - mcl_1st

        mcd_1st = cd_spl_1st.integral(0, 0.5)
        mcd_4th = cd_spl_4th.integral(0, 0.5)
        mcd_wake = mcd_4th - mcd_1st

        mcmh_1st = cmh_spl_1st.integral(0, 0.5)
        mcmh_4th = cmh_spl_4th.integral(0, 0.5)
        mcmh_wake = mcmh_4th - mcmh_1st

        mcmv_1st = cmv_spl_1st.integral(0, 0.5)
        mcmv_4th = cmv_spl_4th.integral(0, 0.5)
        mcmv_wake = mcmv_4th - mcmv_1st

        #---------mid half stroke averages-----------
        mcl_1st_m = cl_spl_1st.integral(0, 0.25)
        mcl_4th_m = cl_spl_4th.integral(0, 0.25)
        mcl_wake_m = mcl_4th_m - mcl_1st_m

        mcd_1st_m = cd_spl_1st.integral(0, 0.25)
        mcd_4th_m = cd_spl_4th.integral(0, 0.25)
        mcd_wake_m = mcd_4th_m - mcd_1st_m

        mcmh_1st_m = cmh_spl_1st.integral(0, 0.25)
        mcmh_4th_m = cmh_spl_4th.integral(0, 0.25)
        mcmh_wake_m = mcmh_4th_m - mcmh_1st_m

        mcmv_1st_m = cmv_spl_1st.integral(0, 0.25)
        mcmv_4th_m = cmv_spl_4th.integral(0, 0.25)
        mcmv_wake_m = mcmv_4th_m - mcmv_1st_m
        #-------------------------------------------------------------

        mcl_1st_arr.append(mcl_1st)
        mcl_4th_arr.append(mcl_4th)
        mcl_wake_arr.append(mcl_wake)
        mcl_wake_mid_arr.append(mcl_wake_m)

        mcd_1st_arr.append(mcd_1st)
        mcd_4th_arr.append(mcd_4th)
        mcd_wake_arr.append(mcd_wake)
        mcd_wake_mid_arr.append(mcd_wake_m)

        mcmh_1st_arr.append(mcmh_1st)
        mcmh_4th_arr.append(mcmh_4th)
        mcmh_wake_arr.append(mcmh_wake)
        mcmh_wake_mid_arr.append(mcmh_wake_m)

        mcmv_1st_arr.append(mcmv_1st)
        mcmv_4th_arr.append(mcmv_4th)
        mcmv_wake_arr.append(mcmv_wake)
        mcmv_wake_mid_arr.append(mcmv_wake_m)

        with open('meanForceCoeffs_' + figureName + '.dat', 'w') as f:
            f.write(
                "case,mcl_1st,mcl_4th,mcl_wake,mcl_wake_mid,mcd_1st,mcd_4th,mcd_wake,mcd_wake_mid,mcmh_1st,mcmh_4th,mcmh_wake,mcmh_wake_mid,mcmv_1st,mcmv_4th,mcmv_wake,mcmv_wake_mid\n"
            )
            for cl1, cl4, clw, clw_m, cd1, cd4, cdw, cdw_m, cmh1, cmh4, cmhw, cmhw_m, cmv1, cmv4, cmvw, cmvw_m, cf_lgd in zip(
                    mcl_1st_arr, mcl_4th_arr, mcl_wake_arr, mcl_wake_mid_arr,
                    mcd_1st_arr, mcd_4th_arr, mcd_wake_arr, mcl_wake_mid_arr,
                    mcmh_1st_arr, mcmh_4th_arr, mcmh_wake_arr,
                    mcmh_wake_mid_arr, mcmv_1st_arr, mcmv_4th_arr,
                    mcmv_wake_arr, mcmv_wake_mid_arr, legends):
                f.write(
                    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" %
                    (cf_lgd, '{0:.8g}'.format(cl1), '{0:.8g}'.format(cl4),
                     '{0:.8g}'.format(clw), '{0:.8g}'.format(clw_m),
                     '{0:.8g}'.format(cd1), '{0:.8g}'.format(cd4),
                     '{0:.8g}'.format(cdw), '{0:.8g}'.format(cdw_m),
                     '{0:.8g}'.format(cmh1), '{0:.8g}'.format(cmh4),
                     '{0:.8g}'.format(cmhw), '{0:.8g}'.format(cmhw_m),
                     '{0:.8g}'.format(cmv1), '{0:.8g}'.format(cmv4),
                     '{0:.8g}'.format(cmvw), '{0:.8g}'.format(cmvw_m)))

    if time_to_plot != 'all':
        axs[0].set_xlim(time_to_plot)
        axs[1].set_xlim(time_to_plot)
        axs[2].set_xlim(time_to_plot)
        axs[3].set_xlim(time_to_plot)
        axs[4].set_xlim(time_to_plot)
    if range_cl != 'all':
        axs[1].set_ylim(range_cl)
        axs[2].set_ylim(range_cd)
        axs[3].set_ylim(range_cmh)
        axs[4].set_ylim(range_cmv)

    for ax in axs:
        ax.axhline(y=0, color='k', linestyle='-.', linewidth=0.5)
        ax.axvline(x=0.5, color='k', linestyle='-.', linewidth=0.5)
        ax.axvline(x=1.5, color='k', linestyle='-.', linewidth=0.5)
        ax.axvline(x=2.5, color='k', linestyle='-.', linewidth=0.5)
        ax.set_xlabel(r'$\^t$')

    axs[0].set_ylabel(r'Kinematics')
    axs[1].set_ylabel(r'$C_L$')
    axs[2].set_ylabel(r'$C_D$')
    axs[3].set_ylabel(r'$Cmh$')
    axs[4].set_ylabel(r'$Cmv$')

    axs[0].legend(loc='upper center',
                  bbox_to_anchor=(legendx, legendy),
                  ncol=2,
                  fontsize='small',
                  frameon=False)
    axs[1].legend(loc='upper center',
                  bbox_to_anchor=(legendx, legendy),
                  ncol=2,
                  fontsize='small',
                  frameon=False)

    title = figureName
    out_image_file = os.path.join(out_dir, title + '.svg')
    fig.savefig(out_image_file)
    # plt.show()

    return fig
