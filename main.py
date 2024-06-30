import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import re


def get_cl(alpha, m):  # m is VERY IMPORTANT TO GET RIGHT
    return alpha * m


def gamma_dist(y, L0, b, rho, Vinf):  # note that this assumes an elliptical lift distribution
    dist = 1 / (rho * Vinf) * L0 * -y / (b ** 2 * np.sqrt(1 - (y / b) ** 2))

    return dist


def discrete_wing(root_chord, root_offset, tip_chord, tip_offset, n):
    le = np.linspace(root_offset, tip_offset, n - 1)
    te = np.linspace(root_offset + root_chord, tip_offset + tip_chord, n - 1)

    chord = te - le

    return chord


def get_induced_alpha(Vinf, g, y):
    dgamma = np.gradient(g, y)

    a_i = np.zeros(len(y))
    for i in range(len(y)):
        np.seterr(divide='ignore')
        f = np.array(dgamma / (y[i] - y))

        nan_arr = np.isinf(f)
        for check_index in range(len(nan_arr)):
            if nan_arr[check_index]:
                if 0 < check_index < len(nan_arr) - 1:
                    f[check_index] = (f[check_index - 1] + f[check_index + 1]) / 2
                elif not check_index:
                    f[check_index] = f[check_index + 1]
                else:
                    f[check_index] = f[check_index - 1]

        a_i[i] = 1 / (4 * np.pi * Vinf) * np.trapz(f, y)

    return a_i


def get_effective_alpha(a, a_i, y):
    a_e = a * np.ones(len(y)) - a_i

    return a_e


def get_new_gamma_dist(Vinf, chord, c_l_dist):
    gamma_iter = 1 / 2 * Vinf * chord * c_l_dist

    return gamma_iter


def compare_gamma(gamma_i, gamma_f):
    return np.linalg.norm(gamma_f - gamma_i)


def get_lift(Vinf, S, g, y):
    return 2 * 2 / (Vinf * S) * np.trapz(g, y)


def get_induced_drag(Vinf, S, g, y):
    a_i = get_induced_alpha(Vinf, g, y)

    return 2 * 2 / (Vinf * S) * np.trapz(g * a_i, y)


def get_Re(rho, u, c, mu):
    return rho * u * c / mu


class clData:
    def __init__(self, Re, Alpha=[], Cl=[]):
        self.Re = Re
        self.Alpha = []
        self.Cl = []

    def fetch(self):
        data_dir = input("Paste path to folder containing cl data:\n")
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
    half_span = 0.750  # [m]
    root_c = 0.309  # [m]
    num_stations = 600

    stations = np.linspace(0, half_span, num_stations)
    stations = stations[:-1]

    aoa = 2  # [deg]
    aoa = aoa * np.ones(num_stations - 1)
    freestream = 9.58 * (200000 / 200000)  # [m/s]
    area = 0.42  # [m2]

    gamma = gamma_dist(stations, 30, half_span, 1.225, freestream)  # don't think L0 does anything

    alpha_i = get_induced_alpha(freestream, gamma, stations)
    alpha_e = get_effective_alpha(aoa, alpha_i, stations)

    Re_list = [50000, 100000, 200000, 500000, 1000000]
    Re_dict = {}
    for i in range(0, len(Re_list)):
        Re_dict[Re_list[i]] = clData(Re_list[i])

    cl_alpha = 0.06

    c_l = get_cl(alpha_e, cl_alpha)

    gamma_new = get_new_gamma_dist(freestream, discrete_wing(root_c, 0, root_c, 0, num_stations), c_l)

    j = 0
    err = [compare_gamma(gamma, gamma_new)]
    D = 0.05
    while (err[j] > 0.01) & (j < 600):
        gamma = gamma + D * (gamma_new - gamma)

        alpha_i = get_induced_alpha(freestream, gamma, stations)
        alpha_e = get_effective_alpha(aoa, alpha_i, stations)
        c_l = get_cl(alpha_e, stations)
        gamma_new = get_new_gamma_dist(freestream, discrete_wing(root_c, 0, root_c, 0, num_stations), c_l)

        j += 1
        err.append(compare_gamma(gamma, gamma_new))

    plt.semilogy(np.linspace(0, j, j + 1), err)
    plt.grid()
    plt.title('L2 error norm of circulation distribution')
    plt.ylabel('L2 error norm')
    plt.xlabel('Iteration')
    plt.show()

    print(f'Wing lift coefficient C_L={round(get_lift(freestream, area, gamma_new, stations), 3)}')
    print(f'Wing induced drag coefficient C_Di={round(get_induced_drag(freestream, area, gamma_new, stations), 6)}')

    test = clData(50000)
    test.fetch()
    print(test.Cl)
