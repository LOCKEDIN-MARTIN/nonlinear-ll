import numpy as np
import matplotlib.pyplot as plt


def get_cl(alpha, y):  # this assumes a NACA0012 airfoil, unstalled behaviour

    m = 0.1454  # assumed (for now) lift slope, HIGHLY SENSITIVE
    c_l = m * alpha

    return c_l


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


if __name__ == '__main__':
    half_span = 0.750  # [m]
    root_c = 0.309  # [m]
    num_stations = 600

    stations = np.linspace(0, half_span, num_stations)
    stations = stations[:-1]

    aoa = 2  # [deg]
    aoa = aoa * np.ones(num_stations - 1)
    freestream = 9.58 * (200000/200000)  # [m/s]
    area = 0.42  # [m2]

    gamma = gamma_dist(stations, 30, half_span, 1.225, freestream)  # don't think L0 does anything

    alpha_i = get_induced_alpha(freestream, gamma, stations)
    alpha_e = get_effective_alpha(aoa, alpha_i, stations)

    c_l = get_cl(alpha_e, stations)

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
