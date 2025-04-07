import numpy as np


def read_transverse_impedance_file(path: str) -> tuple[list[complex]]:
    zp, zm = [], []
    for m, zp_real, zp_imag, zm_real, zm_imag in np.loadtxt(path, comments="#"):
        zp.append(complex(zp_real, -zp_imag))
        zm.append(complex(zm_real, -zm_imag))
    return (zp, zm)


def read_longitudinal_impedance_file(path: str) -> list[complex]:
    z = []
    for real, imag in np.loadtxt(path, comments="#"):
        z.append(complex(real, imag))
    return z