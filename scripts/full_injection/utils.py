def get_intensity(power: float, energy: float, frequency: float) -> float:
    joules_per_ev = 1.6022e-19
    energy_per_pulse = power / frequency / joules_per_ev
    intensity = energy_per_pulse / energy
    return intensity
