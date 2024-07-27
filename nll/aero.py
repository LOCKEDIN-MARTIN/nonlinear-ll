import numpy as np
from numpy.typing import ArrayLike, NDArray


# TODO: Unused?
def get_cl(alpha: float, m):  # m is VERY IMPORTANT TO GET RIGHT
    return alpha * m


def gamma_dist(
    y: ArrayLike, L0: float, b: float, rho: float, Vinf: float
) -> NDArray[np.floating]:
    """
    Returns an elliptical distribution suitable as an initial guess for the
    circulation distribution.

    Parameters
    ----------
    y : ArrayLike
        Array of y-values.
    L0 : float
        Lift of the wing. (apparently doesn't do anything)
    b : float
        Wing half-span.
    rho : float
        Air density.
    Vinf : float
        Freestream velocity.

    Returns
    -------
    NDArray[np.floating]
        The circulation distribution

    """
    dist = 1 / (rho * Vinf) * L0 * -y / (b**2 * np.sqrt(1 - (y / b) ** 2))

    return dist


def get_induced_alpha(Vinf: float, g: ArrayLike, y: ArrayLike) -> NDArray[np.floating]:
    """
    Returns the induced angle of attack at each station.

    Parameters
    ----------
    Vinf : float
        Freestream velocity.
    g : ArrayLike
        Circulation distribution.
    y : ArrayLike
        Array of y-values.

    Returns
    -------
    NDArray[np.floating]
        induced angle of attack at each station
    """

    dgamma = np.gradient(g, y)

    a_i = np.zeros(len(y))
    for i in range(len(y)):
        np.seterr(divide="ignore")
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

        a_i[i] = 1 / (4 * np.pi * Vinf) * np.trapezoid(f, y)

    return a_i


def get_effective_alpha(a: float, a_i: ArrayLike) -> NDArray[np.floating]:
    """
    Returns the effective angle of attack at each station.

    Parameters
    ----------
    a : float
        Angle of attack.
    a_i : ArrayLike
        Induced angle of attack (from circulation effects)

    Returns
    -------
    NDArray[np.floating]
        effective angle of attack at each station
    """
    # it's a trivial operation, but it's here for clarity
    return a - a_i


def get_new_gamma_dist(
    Vinf: float, chord: ArrayLike, c_l_dist: ArrayLike
) -> NDArray[np.floating]:
    """
    Returns the new circulation distribution, updated by cl data.

    Parameters
    ----------
    Vinf : float
        Freestream velocity.
    chord : ArrayLike
        Chord length at each station.
    c_l_dist : ArrayLike
        Lift coefficient distribution.

    Returns
    -------
    NDArray[np.floating]
        Updated circulation distribution
    """
    gamma_iter = 1 / 2 * Vinf * chord * c_l_dist

    return gamma_iter


def get_lift(Vinf: float, S: float, g: ArrayLike, y: ArrayLike) -> float:
    """
    Calculate the lift coefficient (Cl)

    Parameters
    ----------
    Vinf : float
        Freestream velocity.
    S : float
        Wing area.
    g : ArrayLike
        Circulation distribution.
    y : ArrayLike
        Array of y-values.

    Returns
    -------
    float
        Lift coefficient
    """
    return 2 / (Vinf * S) * np.trapezoid(g, y)


def get_induced_drag(C_l, AR, e) -> float:
    """
    Calculate the induced drag coefficient (Cdi)

    Parameters
    ----------
    C_l : float
        Lift coefficient.
    AR : float
        Aspect ratio.
    e : float
        Oswald efficiency factor.

    Returns
    -------
    float
        Induced drag coefficient.
    """
    return C_l**2 / (np.pi * AR * e)


def get_Re(rho: float, u: float, c: float, mu: float) -> float:
    """
    Calculate the Reynolds number for a given flow.

    Parameters
    ----------
    rho : float
        Density [kg/m^3].
    u : float
        Velocity [m/s].
    c : float
        Chord length [m].
    mu : float
        Dynamic viscosity [kg/(m*s)].

    Returns
    -------
    float
        Reynolds number.
    """
    return rho * u * c / mu
