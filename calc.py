import numpy as np


def compare_gamma(gamma_i, gamma_f):
    return np.linalg.norm(gamma_f - gamma_i)


def reduce(clData_list):

    scm = -100  # smallest common minimum
    lcm = 100  # largest common maximum

    for i in clData_list:
        if i.Alpha[0] >= scm:
            scm = i.Alpha[0]
        if i.Alpha[-1] <= lcm:
            lcm = i.Alpha[-1]

    for j in clData_list:

        lb = j.Alpha.index(scm)
        ub = j.Alpha.index(lcm)

        j.Alpha = j.Alpha[lb:ub+1]
        j.Cl = j.Cl[lb:ub+1]
