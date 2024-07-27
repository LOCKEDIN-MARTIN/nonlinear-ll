import numpy as np
from numpy.typing import NDArray


def discrete_wing(
    root_chord: float, root_offset: float, tip_chord: float, tip_offset: float, n: int
) -> NDArray[np.floating]:
    """
    Generates a discrete wing with n stations, given the root and tip chord and
    offsets, returning the chord at each station.

    Parameters
    ----------
    root_chord : float
        The chord of the wing at the root.
    root_offset : float
        x-offset of wing root.
    tip_chord : float
        The chord of the wing at the tip.
    tip_offset : float
        x-offset of wing tip.
    n : int
        The number of stations to generate.

    Returns
    -------
    NDArray[np.floating]
        The chord at each station.

    """
    le = np.linspace(root_offset, tip_offset, n - 1)
    te = np.linspace(root_offset + root_chord, tip_offset + tip_chord, n - 1)

    chord = te - le

    return chord
