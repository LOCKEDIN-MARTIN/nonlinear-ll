import numpy as np
from operator import itemgetter


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

    intr = list(set.intersection(*[set(x.Alpha) for x in clData_list]))
    intr.sort()

    for k in clData_list:

        tempAlpha = k.Alpha
        tempCl = k.Cl

        indices = [k.Alpha.index(x) for x in intr]
        k.Alpha = [tempAlpha[x] for x in indices]
        k.Cl = [tempCl[x] for x in indices]

def simpsons_with_singularity_fix(f_vals, x_vals, x_n, eps=1e-10):
    """
    Approximates integral: a(x_n) = ∫ f(x) / (x_n - x) dx using Simpson's Rule,
    handling singularity at x = x_n by replacing it with the average of neighbors,
    or just the neighbor if it's at the boundary.

    Parameters:
        f_vals: ndarray, values of dG/dx at x_vals
        x_vals: ndarray, grid of x values (must be odd-sized and uniformly spaced)
        x_n: float, the location where the singularity occurs (one of the x_vals)
        eps: float, threshold for detecting singularity

    Returns:
        float, approximated integral
    """

    N = len(x_vals)
    if N % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points")
    
    h = x_vals[1] - x_vals[0]
    integrand = np.zeros(N)

    for i in range(N):
        dx = x_n - x_vals[i]

        if abs(dx) < eps:
            # Singularity detected
            if i == 0:
                # Left boundary: copy from next point
                dx_next = x_n - x_vals[i + 1]
                integrand[i] = f_vals[i + 1] / dx_next
            elif i == N - 1:
                # Right boundary: copy from previous point
                dx_prev = x_n - x_vals[i - 1]
                integrand[i] = f_vals[i - 1] / dx_prev
            else:
                # Interior: average neighbors
                dx_prev = x_n - x_vals[i - 1]
                dx_next = x_n - x_vals[i + 1]
                val_prev = f_vals[i - 1] / dx_prev
                val_next = f_vals[i + 1] / dx_next
                integrand[i] = 0.5 * (val_prev + val_next)
        else:
            integrand[i] = f_vals[i] / dx

    # Simpson’s Rule
    result = integrand[0] + integrand[-1]
    result += 4 * np.sum(integrand[1:-1:2])  # odd indices
    result += 2 * np.sum(integrand[2:-2:2])  # even indices
    result *= h / 3

    return result