import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import re
import scipy

import aero
import geom
import calc


class clData:
    def __init__(self, Re):
        self.Re = Re
        self.Alpha = []
        self.Cl = []
        self.Cl_Alpha = []

    def fetch(self, data_dir):

        file_prefix = r'\xf-n0012-il-'
        file_suffix = '.csv'

        target = Path(data_dir + file_prefix + str(self.Re) + file_suffix)

        if not target.is_file():
            print('File not found, double-check directory and name of: ', target)

        else:
            with open(target, newline='') as csvfile:
                reader = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))  # converted to list for indexing
            csvfile.close()

            blank_line = 0
            row_index = 0
            while not blank_line:
                if not reader[row_index]:
                    blank_line = 1
                else:
                    row_index += 1

            reader = reader[row_index + 1:]
            reformatted_reader = []
            for row in reader:
                string_to_split = row[0]
                reformatted_reader.append(re.split(',', string_to_split))

            var_name = reformatted_reader[0]
            alpha_index = var_name.index('Alpha')
            cl_index = var_name.index('Cl')

            for row_index in range(1, len(reformatted_reader)):
                reformatted_reader[row_index] = [float(ii) for ii in reformatted_reader[row_index]]
                curr = reformatted_reader[row_index]
                self.Alpha.append(curr[alpha_index])
                self.Cl.append(curr[cl_index])


if __name__ == '__main__':
    half_span = 0.724  # [m]
    root_c = 0.4  # [m]
    tip_c = 0.234  # [m]
    tip_offset = 0.086  # [m]
    num_stations = 7

    aspect_ratio = 4.5
    eff = 0.98

    freestream = 10  # [m/s]
    area = 0.233  # [m2]

    chord = geom.discrete_wing(root_c, 0, root_c, tip_offset, num_stations)
    Re_stations = aero.get_Re(1.225, freestream, chord, 0.00001837)

    stations = np.linspace(0, half_span, num_stations)
    stations = stations[:-1]

    data_50k = clData(50000)
    data_100k = clData(100000)
    data_200k = clData(200000)
    data_500k = clData(500000)
    data_1m = clData(1000000)

    data_dir = input("Paste path to folder containing cl data:\n")

    data_50k.fetch(data_dir)
    data_100k.fetch(data_dir)
    data_200k.fetch(data_dir)
    data_500k.fetch(data_dir)
    data_1m.fetch(data_dir)

    Re_list = [50000, 100000, 200000, 500000, 1000000]
    Re_dict = {}
    for i in range(0, len(Re_list)):
        Re_dict[Re_list[i]] = clData(Re_list[i])

    calc.reduce([data_50k, data_100k, data_200k, data_500k, data_1m])
    data = [data_50k, data_100k, data_200k, data_500k, data_1m]

    num_angles = 35
    alpha_sweep = np.linspace(min(data[0].Alpha), max(data[0].Alpha), num_angles)

    c_l_sweep = []
    c_di_sweep = []

    z = []
    for i in data:
        z += i.Cl

    y = []
    for j in Re_list:
        k = 0
        while k < len(data[0].Alpha):
            y.append(j)
            k += 1

    x = []
    for i in Re_list:
        x += data[0].Alpha

    coords = list(zip(x, y))
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for aoa in alpha_sweep:

        aoa = aoa * np.ones(num_stations - 1)

        gamma = aero.gamma_dist(stations, 30, half_span, 1.225, freestream)  # don't think L0 does anything

        alpha_i = aero.get_induced_alpha(freestream, gamma, stations)
        alpha_e = aero.get_effective_alpha(aoa, alpha_i, stations)

        c_l = scipy.interpolate.LinearNDInterpolator(coords, z)  # fix this
        A, R = np.meshgrid(data[0].Alpha, Re_list)
        cl_function = c_l(A, R)
        cl_interpolated = c_l.__call__(alpha_e, Re_stations)

        gamma_new = aero.get_new_gamma_dist(freestream, chord, cl_interpolated)

        j = 0
        err = [calc.compare_gamma(gamma, gamma_new)]
        D = 0.05
        while (err[j] > 0.01) & (j < 600):
            gamma = gamma + D * (gamma_new - gamma)

            alpha_i = aero.get_induced_alpha(freestream, gamma, stations)
            alpha_e = aero.get_effective_alpha(aoa, alpha_i, stations)

            cl_interpolated = c_l.__call__(alpha_e, Re_stations)
            gamma_new = aero.get_new_gamma_dist(freestream, chord, cl_interpolated)

            j += 1
            err.append(calc.compare_gamma(gamma, gamma_new))

        c_l_sweep.append(aero.get_lift(freestream, area, gamma_new, stations))
        c_di_sweep.append(aero.get_induced_drag(aero.get_lift(freestream, area, gamma_new, stations), aspect_ratio, eff))

    ax1.plot(alpha_sweep, c_l_sweep)
    ax2.plot(alpha_sweep, c_di_sweep)

    ax1.set_title('Lift coefficient')
    ax2.set_title('Induced drag coefficient')
    ax1.set(ylabel='C_l')
    ax2.set(ylabel='C_di')
    fig.text(0.5, 0.04, 'Alpha [deg]', ha='center')
    ax1.grid()
    ax2.grid()
    plt.show()

    fields = ['Alpha', 'C_di', 'C_l']

    predict = []
    for k in range(len(alpha_sweep)):
        predict.append([str(alpha_sweep[k]), str(c_di_sweep[k]), str(c_l_sweep[k])])

    writename = 'output.csv'

    with open(writename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)
        writer.writerows(predict)

    csvfile.close()
