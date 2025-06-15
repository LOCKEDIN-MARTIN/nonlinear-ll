import os
import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

data_dir = r"C:\Users\Daniel F\Documents\GitHub\nonlinear-ll\n0012_xfoil_data\\"
pattern = re.compile(r"xf-n0012-il-(\d+)\.csv")

re_to_cl_interp = {}
alpha_union = []

# find all alphas
for filename in os.listdir(data_dir):
    match = pattern.match(filename)
    if not match:
        continue

    Re = int(match.group(1))
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, skiprows=10)
    df.columns = df.columns.str.strip()

    alpha = df['Alpha'].to_numpy()
    alpha_union.extend(alpha)

# Build sorted, unique alpha grid
alpha_common = np.linspace(min(alpha_union), max(alpha_union), 200)

# interpolate Cl to common alpha
cl_data = {}
Re_values = []

for filename in os.listdir(data_dir):
    match = pattern.match(filename)
    if not match:
        continue

    Re = int(match.group(1))
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, skiprows=10)
    df.columns = df.columns.str.strip()

    alpha = df['Alpha'].to_numpy()
    cl = df['Cl'].to_numpy()

    # Interpolate Cl onto the common alpha grid
    interp_func = interp1d(alpha, cl, bounds_error=False, fill_value=np.nan)
    cl_interp = interp_func(alpha_common)

    cl_data[Re] = cl_interp
    Re_values.append(Re)

Re_values = sorted(Re_values)
cl_grid = np.stack([cl_data[Re] for Re in Re_values], axis=1)  # shape: (n_alpha, n_Re)
Re_grid = np.array(Re_values)

interpolator = RegularGridInterpolator((alpha_common, Re_grid), cl_grid, bounds_error=False, fill_value=np.nan)

print(interpolator((6, 60000)))