import numpy as np

def diffusion3D(r, a):
    return 1 / ((1+r)*np.sqrt(1+r/a**2))

def triplet_decay(r, F):
    return (1 - F + F*np.exp(-r)) / (1 - F)

def triplet_diffusion(t: np.ndarray, N, g_inf, T_trip, F_trip, T1, a):
    """Triplet state + diffusion"""
    return triplet_decay(t/T_trip, F_trip)*diffusion3D(t/T1, a)/N + g_inf