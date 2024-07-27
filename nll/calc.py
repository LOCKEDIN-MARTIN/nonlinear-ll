from itertools import chain

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from nll.aerodata import CLData


def compare_gamma(gamma_i, gamma_f):
    return np.linalg.norm(gamma_f - gamma_i)


def reduce(data_list: list[CLData]):
    """
    Reduces the data to the intersection of the alpha values in all the data sets.

    Modifies the datasets in place.

    Parameters
    ----------
    data_list : list[CLData]
        The list of datasets to reduce.
    """
    # find only alphas which are common to all datasets
    all_alphas = [set(x.Alpha) for x in data_list]

    common_alpha = sorted(set.intersection(*all_alphas))

    for cld in data_list:
        tempAlpha = cld.Alpha
        tempCl = cld.Cl

        indices = [cld.Alpha.index(x) for x in common_alpha]
        cld.Alpha = [tempAlpha[x] for x in indices]
        cld.Cl = [tempCl[x] for x in indices]


def generate_aerodata_interpolant(
    data_list: list[CLData],
) -> LinearNDInterpolator:
    """
    Generate an interpolant for the aerodynamic data.

    Parameters
    ----------
    data_list : list[CLData]
        The list of datasets to interpolate, should be reduced first.

    Returns
    -------
    Callable[[float, float], float]
        The interpolant function.
    """
    # construct interpolant
    # output variable: Cl; input variables: Alpha, Re

    # join all the data lists "end-to-end
    v_cl = list(chain.from_iterable(d.Cl for d in data_list))
    v_alpha = list(chain.from_iterable(d.Alpha for d in data_list))
    # repeat Re values for each alpha
    v_re = list(chain.from_iterable([d.Re] * len(d.Alpha) for d in data_list))

    coords = list(zip(v_alpha, v_re))
    cl_interpolant = LinearNDInterpolator(coords, v_cl)

    return cl_interpolant
