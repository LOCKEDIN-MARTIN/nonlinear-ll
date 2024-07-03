import numpy as np

def compare_gamma(gamma_i, gamma_f):
    return np.linalg.norm(gamma_f - gamma_i)