import argparse

import matplotlib.pyplot as plt
import numpy as np

from nll import aero, calc, compute, geom
from nll.aerodata import CLData

if __name__ == "__main__":
    # constants
    FILE_PREFIX = "xf-n0012-il-"
    NUM_ANGLES = 35

    # program arguments
    parser = argparse.ArgumentParser(
        description="Calculate lift and drag coefficients for a finite wing"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to folder containing aero data",
        default="n0012_xfoil_data",
        required=False,
    )

    args = parser.parse_args()

    # aircraft parameters
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

    stations = np.linspace(0, half_span, num_stations)
    stations = stations[:-1]

    # Load Data
    Re_list = [50000, 100000, 200000, 500000, 1000000]

    # (use a dictionary comprehension for compactness)
    Re_dict = {r: CLData.from_file(r, args.data_dir, FILE_PREFIX) for r in Re_list}
    data = list(Re_dict.values())
    calc.reduce(data)  # reduce data to common alpha values

    # Prepare interpolant
    cl_interpolant = calc.generate_aerodata_interpolant(data)

    # Compute
    alpha_sweep = np.linspace(min(data[0].Alpha), max(data[0].Alpha), NUM_ANGLES)
    gamma = compute.calculate_circulation(
        cl_interpolant, alpha_sweep, stations, chord, freestream, half_span
    )
    c_l_sweep = np.zeros_like(alpha_sweep)
    c_di_sweep = np.zeros_like(alpha_sweep)

    for idx, g_dist in enumerate(gamma):
        resultant_lift = aero.get_lift(freestream, area, g_dist, stations)
        resultant_drag = aero.get_induced_drag(resultant_lift, aspect_ratio, eff)

        c_l_sweep[idx] = resultant_lift
        c_di_sweep[idx] = resultant_drag

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(alpha_sweep, c_l_sweep)
    ax2.plot(alpha_sweep, c_di_sweep)

    ax1.set_title("Lift coefficient")
    ax2.set_title("Induced drag coefficient")
    ax1.set(ylabel="C_l")
    ax2.set(ylabel="C_di")
    fig.text(0.5, 0.04, "Alpha [deg]", ha="center")
    ax1.grid()
    ax2.grid()
    plt.show()

    fields = ["Alpha", "C_di", "C_l"]

    # use numpy's array exporter because it is simple and efficient
    export_data_array = np.array([alpha_sweep, c_di_sweep, c_l_sweep]).T
    np.savetxt(
        "output.csv",
        export_data_array,
        delimiter=",",
        header=",".join(fields),
        fmt="%1.6f",  # adjust precision if needed (default is 6)
    )
