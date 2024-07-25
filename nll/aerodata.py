from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CLData:
    """
    Abstract class for storing and fetching cl data from csv files
    """

    Re: float
    Alpha: list[float]
    Cl: list[float]

    @classmethod
    def from_file(cls, Re: float, data_dir: str, file_prefix: str) -> "CLData":
        """
        Create a new instance of the class by fetching data from a file

        Parameters
        ----------
        Re : float
            Reynolds number of the data to be fetched
        data_dir : str
            Directory containing the data file
        file_prefix : str
            Prefix of the data file e.g. "xf-n0012-il-"

        Returns
        -------
        CLData
            Instance of the class with data fetched from the file
        """
        data_file = Path(data_dir) / (file_prefix + str(Re) + ".csv")

        if not data_file.is_file():
            raise FileNotFoundError(
                f"File not found, double-check directory and name of: {data_file}"
            )

        else:
            # read using numpy, more efficient than CSV
            array_data = np.genfromtxt(
                data_file, delimiter=",", dtype=None, skip_header=11, usecols=(0, 1)
            )

            Alpha = list(array_data[:, 0])
            Cl = list(array_data[:, 1])

            return cls(Re, Alpha, Cl)

    @property
    def Cl_Alpha(self) -> np.ndarray:
        # explicit gradient calculation for Cl-Alpha, ensures that changes in alpha
        # are reflected in the lift slope
        return np.gradient(self.Cl, self.Alpha)

    def plot(self, option: str):
        plt.clf()

        if option == "a":
            plt.plot(self.Alpha, self.Cl)
            plt.grid()
            plt.title("Cl-Alpha plot")
            plt.ylabel("Cl")
            plt.xlabel("Alpha")

        elif option == "s":
            plt.plot(self.Alpha, self.Cl_Alpha)
            plt.grid()
            plt.title("Cl-Alpha Slope plot")
            plt.ylabel("Cl-Alpha Slope")
            plt.xlabel("Alpha")

        plt.show()
