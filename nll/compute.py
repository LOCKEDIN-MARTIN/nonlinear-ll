import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm

from nll import aero, calc

# constants
RHO = 1.225  # [kg/m^3]
MU = 0.00001837  # [kg/m/s]


def calculate_circulation(
    cl_interpolant: LinearNDInterpolator,
    alpha_values: ArrayLike,
    y_stations: ArrayLike,
    chord_stations: ArrayLike,
    freestream: float,
    half_span: float,
    conv_tol: float = 0.01,
    max_iter: int = 600,
) -> NDArray[np.floating]:
    """
    Compute the lift and drag coefficients for a given set of alpha values,
    using the nonlinear lifting line method.

    Parameters
    ----------
    cl_interpolant : LinearNDInterpolator
        The interpolant for the aerodynamic data.
    alpha_values : ArrayLike
        The angles of attack to compute the lift and drag coefficients for.
    y_stations : ArrayLike
        The y-values of the stations.
    chord_stations : ArrayLike
        The chord lengths of the stations.
    freestream : float
        The freestream velocity.
    half_span : float
        The half-span of the wing.
    conv_tol : float, optional
        The convergence tolerance, by default 0.01.
    max_iter : int, optional
        The maximum number of iterations, by default 600.

    Returns
    -------
    NDArray[np.floating]
        The circulation distribution at each alpha value.
    """
    Re_stations = aero.get_Re(RHO, freestream, chord_stations, MU)

    # Compute
    gamma_result = np.zeros((len(alpha_values), len(y_stations)))

    for idx, alpha in enumerate(
        tqdm(alpha_values, desc="Calculating circulation distribustions")
    ):
        # initial guess for gamma distribution
        gamma = aero.gamma_dist(
            y_stations, 30, half_span, RHO, freestream
        )  # don't think L0 does anything

        D = 0.05

        # iteration to within an error tolerance
        for _ in range(max_iter):
            alpha_i = aero.get_induced_alpha(freestream, gamma, y_stations)
            alpha_e = aero.get_effective_alpha(alpha, alpha_i)

            cl_interpolated = cl_interpolant(alpha_e, Re_stations)
            gamma_new = aero.get_new_gamma_dist(
                freestream, chord_stations, cl_interpolated
            )

            gamma = gamma + D * (gamma_new - gamma)

            err = calc.compare_gamma(gamma, gamma_new)

            if err < conv_tol:
                break

        gamma_result[idx] = gamma

    return gamma_result
