import numpy as np


def get_intensity(power: float, energy: float, frequency: float) -> float:
    joules_per_ev = 1.6022e-19
    energy_per_pulse = power / frequency / joules_per_ev
    intensity = energy_per_pulse / energy
    return intensity
    
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