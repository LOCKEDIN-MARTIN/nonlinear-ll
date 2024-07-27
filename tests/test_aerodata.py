from pathlib import Path

import numpy as np
import pytest

from nll import aero, calc, geom
from nll.aerodata import CLData


def test_validation_baseline():
    """
    Test using the old __main__ code from aerodata.py
    """
    half_span = 0.724  # [m]
    root_c = 0.4  # [m]
    tip_c = 0.234  # [m]
    tip_offset = 0.086  # [m]
    num_stations = 7

    aspect_ratio = 4.5
    eff = 0.98

    stations = np.linspace(0, half_span, num_stations)
    stations = stations[:-1]

    data_path = Path(__file__).parent / "validation_data.csv"
    arr = np.genfromtxt(data_path, delimiter=",", skip_header=1, usecols=(3, 7))

    aerodata = CLData(100000, arr[:, 1], arr[:, 0])

    num_angles = 35
    alpha_sweep = np.linspace(0, 45, num_angles)

    c_l_sweep = []
    c_di_sweep = []

    for aoa in alpha_sweep:
        aoa = aoa * np.ones(num_stations - 1)
        freestream = 10  # [m/s]
        area = 0.233  # [m2]

        gamma = aero.gamma_dist(
            stations, 30, half_span, 1.225, freestream
        )  # don't think L0 does anything

        alpha_i = aero.get_induced_alpha(freestream, gamma, stations)
        alpha_e = aero.get_effective_alpha(aoa, alpha_i)

        c_l = np.interp(alpha_e, aerodata.Alpha, aerodata.Cl)

        gamma_new = aero.get_new_gamma_dist(
            freestream,
            geom.discrete_wing(root_c, 0, root_c, tip_offset, num_stations),
            c_l,
        )

        j = 0
        err = [calc.compare_gamma(gamma, gamma_new)]
        D = 0.05
        while (err[j] > 0.01) & (j < 600):
            gamma = gamma + D * (gamma_new - gamma)

            alpha_i = aero.get_induced_alpha(freestream, gamma, stations)
            alpha_e = aero.get_effective_alpha(aoa, alpha_i)
            c_l = np.interp(alpha_e, aerodata.Alpha, aerodata.Cl)
            gamma_new = aero.get_new_gamma_dist(
                freestream,
                geom.discrete_wing(root_c, 0, root_c, tip_offset, num_stations),
                c_l,
            )

            j += 1
            err.append(calc.compare_gamma(gamma, gamma_new))

        c_l_sweep.append(aero.get_lift(freestream, area, gamma_new, stations))
        c_di_sweep.append(
            aero.get_induced_drag(
                aero.get_lift(freestream, area, gamma_new, stations), aspect_ratio, eff
            )
        )

    validation_output = np.genfromtxt(
        Path(__file__).parent / "validation_output.csv", delimiter=",", skip_header=1
    )

    val_alpha = validation_output[:, 0]
    val_cl = validation_output[:, 2]
    val_cd = validation_output[:, 1]

    assert alpha_sweep == pytest.approx(val_alpha, rel=1e-6)
    assert np.array(c_l_sweep) == pytest.approx(val_cl, rel=1e-6)
    assert np.array(c_di_sweep) == pytest.approx(val_cd, rel=1e-6)
