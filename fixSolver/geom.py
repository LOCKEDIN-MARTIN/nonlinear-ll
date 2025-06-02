import numpy as np


def discrete_wing(root_chord, root_offset, tip_chord, tip_offset, n):
    le = np.linspace(root_offset, tip_offset, n - 1)
    te = np.linspace(root_offset + root_chord, tip_offset + tip_chord, n - 1)

    chord = te - le

    return chord