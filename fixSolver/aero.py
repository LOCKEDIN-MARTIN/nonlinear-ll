import numpy as np


def get_cl(alpha, m):  # m is VERY IMPORTANT TO GET RIGHT
    return alpha * m


def gamma_dist(y, L0, b, rho, Vinf):  # note that this assumes an elliptical lift distribution
    dist = 1 / (rho * Vinf) * L0 * -y / (b ** 2 * np.sqrt(1 - (y / b) ** 2))

    return dist


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


def get_lift(Vinf, S, g, y):
    return 2 / (Vinf * S) * np.trapz(g, y)


def get_induced_drag(C_l, AR, e):
    return C_l ** 2 / (np.pi * AR * e)


def get_Re(rho, u, c, mu):
    return rho * u * c / mu