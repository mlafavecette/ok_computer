import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import h5py
import json
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.local_env import VoronoiNN
from ase.io import read, write
import random
from scipy.stats import truncnorm
import os


class MaterialsDataGenerator:
    """Comprehensive materials data generator for ML training"""

    def __init__(self, batch_size=32, seed=42):
        self.batch_size = batch_size
        self.seed = seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Load configurations
        self.elements = self._load_elements()
        self.property_ranges = self._load_property_ranges()
        self.element_features = self._load_element_features()

        # Initialize network-specific parameters
        self.max_atoms = 150
        self.max_neighbors = 12
        self.cutoff_radius = 5.0  # Angstroms

        # Setup data generation parameters
        self.temperature_range = (100, 1000)  # K
        self.pressure_range = (1, 100)  # atm
        self.composition_range = (2, 5)  # min/max elements in compound

    def _load_elements(self) -> Dict[str, Dict]:
        elements = {
            'H': {'Z': 1, 'mass': 1.008, 'radius': 0.53, 'electronegativity': 2.20, 'ionization_energy': 13.598,
                  'electron_affinity': 0.754, 'oxidation_states': [-1, 1]},
            'He': {'Z': 2, 'mass': 4.003, 'radius': 0.31, 'electronegativity': 0.00, 'ionization_energy': 24.587,
                   'electron_affinity': 0.000, 'oxidation_states': [0]},
            'Li': {'Z': 3, 'mass': 6.941, 'radius': 1.67, 'electronegativity': 0.98, 'ionization_energy': 5.392,
                   'electron_affinity': 0.618, 'oxidation_states': [1]},
            'Be': {'Z': 4, 'mass': 9.012, 'radius': 1.12, 'electronegativity': 1.57, 'ionization_energy': 9.323,
                   'electron_affinity': 0.000, 'oxidation_states': [2]},
            'B': {'Z': 5, 'mass': 10.811, 'radius': 0.87, 'electronegativity': 2.04, 'ionization_energy': 8.298,
                  'electron_affinity': 0.277, 'oxidation_states': [3]},
            'C': {'Z': 6, 'mass': 12.011, 'radius': 0.67, 'electronegativity': 2.55, 'ionization_energy': 11.260,
                  'electron_affinity': 1.263, 'oxidation_states': [-4, -3, -2, -1, 1, 2, 3, 4]},
            'N': {'Z': 7, 'mass': 14.007, 'radius': 0.56, 'electronegativity': 3.04, 'ionization_energy': 14.534,
                  'electron_affinity': 0.000, 'oxidation_states': [-3, -2, -1, 1, 2, 3, 4, 5]},
            'O': {'Z': 8, 'mass': 15.999, 'radius': 0.48, 'electronegativity': 3.44, 'ionization_energy': 13.618,
                  'electron_affinity': 1.461, 'oxidation_states': [-2, -1, 1, 2]},
            'F': {'Z': 9, 'mass': 18.998, 'radius': 0.42, 'electronegativity': 3.98, 'ionization_energy': 17.423,
                  'electron_affinity': 3.339, 'oxidation_states': [-1]},
            'Ne': {'Z': 10, 'mass': 20.180, 'radius': 0.38, 'electronegativity': 0.00, 'ionization_energy': 21.565,
                   'electron_affinity': 0.000, 'oxidation_states': [0]},
            'Na': {'Z': 11, 'mass': 22.990, 'radius': 1.90, 'electronegativity': 0.93, 'ionization_energy': 5.139,
                   'electron_affinity': 0.548, 'oxidation_states': [1]},
            'Mg': {'Z': 12, 'mass': 24.305, 'radius': 1.45, 'electronegativity': 1.31, 'ionization_energy': 7.646,
                   'electron_affinity': 0.000, 'oxidation_states': [2]},
            'Al': {'Z': 13, 'mass': 26.982, 'radius': 1.18, 'electronegativity': 1.61, 'ionization_energy': 5.986,
                   'electron_affinity': 0.441, 'oxidation_states': [3]},
            'Si': {'Z': 14, 'mass': 28.086, 'radius': 1.11, 'electronegativity': 1.90, 'ionization_energy': 8.152,
                   'electron_affinity': 1.385, 'oxidation_states': [-4, -3, -2, -1, 1, 2, 3, 4]},
            'P': {'Z': 15, 'mass': 30.974, 'radius': 0.98, 'electronegativity': 2.19, 'ionization_energy': 10.487,
                  'electron_affinity': 0.746, 'oxidation_states': [-3, -2, -1, 1, 2, 3, 4, 5]},
            'S': {'Z': 16, 'mass': 32.065, 'radius': 0.88, 'electronegativity': 2.58, 'ionization_energy': 10.360,
                  'electron_affinity': 2.077, 'oxidation_states': [-2, -1, 1, 2, 3, 4, 5, 6]},
            'Cl': {'Z': 17, 'mass': 35.453, 'radius': 0.79, 'electronegativity': 3.16, 'ionization_energy': 12.968,
                   'electron_affinity': 3.613, 'oxidation_states': [-1, 1, 3, 5, 7]},
            'Ar': {'Z': 18, 'mass': 39.948, 'radius': 0.71, 'electronegativity': 0.00, 'ionization_energy': 15.760,
                   'electron_affinity': 0.000, 'oxidation_states': [0]},
            'K': {'Z': 19, 'mass': 39.098, 'radius': 2.43, 'electronegativity': 0.82, 'ionization_energy': 4.341,
                  'electron_affinity': 0.501, 'oxidation_states': [1]},
            'Ca': {'Z': 20, 'mass': 40.078, 'radius': 1.94, 'electronegativity': 1.00, 'ionization_energy': 6.113,
                   'electron_affinity': 0.025, 'oxidation_states': [2]},
            'Sc': {'Z': 21, 'mass': 44.956, 'radius': 1.84, 'electronegativity': 1.36, 'ionization_energy': 6.562,
                   'electron_affinity': 0.188, 'oxidation_states': [3]},
            'Ti': {'Z': 22, 'mass': 47.867, 'radius': 1.76, 'electronegativity': 1.54, 'ionization_energy': 6.828,
                   'electron_affinity': 0.084, 'oxidation_states': [-1, 2, 3, 4]},
            'V': {'Z': 23, 'mass': 50.942, 'radius': 1.71, 'electronegativity': 1.63, 'ionization_energy': 6.746,
                  'electron_affinity': 0.525, 'oxidation_states': [-1, 2, 3, 4, 5]},
            'Cr': {'Z': 24, 'mass': 51.996, 'radius': 1.66, 'electronegativity': 1.66, 'ionization_energy': 6.767,
                   'electron_affinity': 0.666, 'oxidation_states': [-2, -1, 1, 2, 3, 4, 5, 6]},
            'Mn': {'Z': 25, 'mass': 54.938, 'radius': 1.61, 'electronegativity': 1.55, 'ionization_energy': 7.434,
                   'electron_affinity': 0.000, 'oxidation_states': [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7]},
            'Fe': {'Z': 26, 'mass': 55.845, 'radius': 1.56, 'electronegativity': 1.83, 'ionization_energy': 7.902,
                   'electron_affinity': 0.163, 'oxidation_states': [-2, -1, 1, 2, 3, 4, 5, 6]},
            'Co': {'Z': 27, 'mass': 58.933, 'radius': 1.52, 'electronegativity': 1.88, 'ionization_energy': 7.881,
                   'electron_affinity': 0.662, 'oxidation_states': [-1, 1, 2, 3, 4, 5]},
            'Ni': {'Z': 28, 'mass': 58.693, 'radius': 1.49, 'electronegativity': 1.91, 'ionization_energy': 7.640,
                   'electron_affinity': 1.156, 'oxidation_states': [-1, 1, 2, 3, 4]},
            'Cu': {'Z': 29, 'mass': 63.546, 'radius': 1.45, 'electronegativity': 1.90, 'ionization_energy': 7.726,
                   'electron_affinity': 1.235, 'oxidation_states': [1, 2, 3, 4]},
            'Zn': {'Z': 30, 'mass': 65.380, 'radius': 1.42, 'electronegativity': 1.65, 'ionization_energy': 9.394,
                   'electron_affinity': 0.000, 'oxidation_states': [2]},
            'Ga': {'Z': 31, 'mass': 69.723, 'radius': 1.36, 'electronegativity': 1.81, 'ionization_energy': 5.999,
                   'electron_affinity': 0.300, 'oxidation_states': [1, 2, 3]},
            'Ge': {'Z': 32, 'mass': 72.640, 'radius': 1.25, 'electronegativity': 2.01, 'ionization_energy': 7.899,
                   'electron_affinity': 1.233, 'oxidation_states': [-4, 1, 2, 3, 4]},
            'As': {'Z': 33, 'mass': 74.922, 'radius': 1.14, 'electronegativity': 2.18, 'ionization_energy': 9.789,
                   'electron_affinity': 0.804, 'oxidation_states': [-3, 2, 3, 5]},
            'Se': {'Z': 34, 'mass': 78.960, 'radius': 1.03, 'electronegativity': 2.55, 'ionization_energy': 9.752,
                   'electron_affinity': 2.021, 'oxidation_states': [-2, 2, 4, 6]},
            'Br': {'Z': 35, 'mass': 79.904, 'radius': 0.94, 'electronegativity': 2.96, 'ionization_energy': 11.814,
                   'electron_affinity': 3.363, 'oxidation_states': [-1, 1, 3, 5]},
            'Kr': {'Z': 36, 'mass': 83.798, 'radius': 0.88, 'electronegativity': 3.00, 'ionization_energy': 14.000,
                   'electron_affinity': 0.000, 'oxidation_states': [0, 2]},
            'Rb': {'Z': 37, 'mass': 85.468, 'radius': 2.65, 'electronegativity': 0.82, 'ionization_energy': 4.177,
                   'electron_affinity': 0.486, 'oxidation_states': [1]},
            'Sr': {'Z': 38, 'mass': 87.620, 'radius': 2.19, 'electronegativity': 0.95, 'ionization_energy': 5.695,
                   'electron_affinity': 0.050, 'oxidation_states': [2]},
            'Y': {'Z': 39, 'mass': 88.906, 'radius': 2.12, 'electronegativity': 1.22, 'ionization_energy': 6.217,
                  'electron_affinity': 0.307, 'oxidation_states': [3]},
            'Zr': {'Z': 40, 'mass': 91.224, 'radius': 2.06, 'electronegativity': 1.33, 'ionization_energy': 6.634,
                   'electron_affinity': 0.426, 'oxidation_states': [-2, 1, 2, 3, 4]},
            'Nb': {'Z': 41, 'mass': 92.906, 'radius': 1.98, 'electronegativity': 1.60, 'ionization_energy': 6.759,
                   'electron_affinity': 0.893, 'oxidation_states': [-1, 2, 3, 4, 5]},
            'Mo': {'Z': 42, 'mass': 95.960, 'radius': 1.90, 'electronegativity': 2.16, 'ionization_energy': 7.092,
                   'electron_affinity': 0.746, 'oxidation_states': [-2, -1, 1, 2, 3, 4, 5, 6]},
            'Tc': {'Z': 43, 'mass': 98.000, 'radius': 1.83, 'electronegativity': 1.90, 'ionization_energy': 7.280,
                   'electron_affinity': 0.550, 'oxidation_states': [-3, -1, 1, 2, 3, 4, 5, 6, 7]},
            'Ru': {'Z': 44, 'mass': 101.070, 'radius': 1.78, 'electronegativity': 2.20, 'ionization_energy': 7.361,
                   'electron_affinity': 1.050, 'oxidation_states': [-2, 1, 2, 3, 4, 5, 6, 7, 8]},
            'Rh': {'Z': 45, 'mass': 102.906, 'radius': 1.73, 'electronegativity': 2.28, 'ionization_energy': 7.459,
                   'electron_affinity': 1.137, 'oxidation_states': [-1, 1, 2, 3, 4, 5, 6]},
            'Pd': {'Z': 46, 'mass': 106.42, 'radius': 1.69, 'electronegativity': 2.20, 'ionization_energy': 8.337,
                   'electron_affinity': 0.562, 'oxidation_states': [0, 2, 4]},
            'Ag': {'Z': 47, 'mass': 107.868, 'radius': 1.65, 'electronegativity': 1.93, 'ionization_energy': 7.576,
                   'electron_affinity': 1.302, 'oxidation_states': [1, 2, 3]},
            'Cd': {'Z': 48, 'mass': 112.411, 'radius': 1.61, 'electronegativity': 1.69, 'ionization_energy': 8.994,
                   'electron_affinity': 0.000, 'oxidation_states': [2]},
            'In': {'Z': 49, 'mass': 114.818, 'radius': 1.56, 'electronegativity': 1.78, 'ionization_energy': 5.786,
                   'electron_affinity': 0.300, 'oxidation_states': [1, 2, 3]},
            'Sn': {'Z': 50, 'mass': 118.710, 'radius': 1.45, 'electronegativity': 1.96, 'ionization_energy': 7.344,
                   'electron_affinity': 1.112, 'oxidation_states': [-4, 2, 4]},
            'Sb': {'Z': 51, 'mass': 121.760, 'radius': 1.33, 'electronegativity': 2.05, 'ionization_energy': 8.640,
                   'electron_affinity': 1.047, 'oxidation_states': [-3, 3, 5]},
            'Te': {'Z': 52, 'mass': 127.600, 'radius': 1.23, 'electronegativity': 2.10, 'ionization_energy': 9.010,
                   'electron_affinity': 1.971, 'oxidation_states': [-2, 2, 4, 6]},
            'I': {'Z': 53, 'mass': 126.904, 'radius': 1.15, 'electronegativity': 2.66, 'ionization_energy': 10.451,
                  'electron_affinity': 3.059, 'oxidation_states': [-1, 1, 3, 5, 7]},
            'Xe': {'Z': 54, 'mass': 131.293, 'radius': 1.08, 'electronegativity': 2.60, 'ionization_energy': 12.130,
                   'electron_affinity': 0.000, 'oxidation_states': [0, 2, 4, 6, 8]},
            'Cs': {'Z': 55, 'mass': 132.905, 'radius': 2.98, 'electronegativity': 0.79, 'ionization_energy': 3.894,
                   'electron_affinity': 0.472, 'oxidation_states': [1]},
            'Ba': {'Z': 56, 'mass': 137.327, 'radius': 2.53, 'electronegativity': 0.89, 'ionization_energy': 5.212,
                   'electron_affinity': 0.145, 'oxidation_states': [2]},
            'La': {'Z': 57, 'mass': 138.905, 'radius': 2.15, 'electronegativity': 1.10, 'ionization_energy': 5.577,
                   'electron_affinity': 0.470, 'oxidation_states': [3]},
            'Ce': {'Z': 58, 'mass': 140.116, 'radius': 2.07, 'electronegativity': 1.12, 'ionization_energy': 5.539,
                   'electron_affinity': 0.500, 'oxidation_states': [2, 3, 4]},
            'Pr': {'Z': 59, 'mass': 140.908, 'radius': 2.06, 'electronegativity': 1.13, 'ionization_energy': 5.473,
                   'electron_affinity': 0.500, 'oxidation_states': [2, 3, 4]},
            'Nd': {'Z': 60, 'mass': 144.242, 'radius': 2.05, 'electronegativity': 1.14, 'ionization_energy': 5.525,
                   'electron_affinity': 0.500, 'oxidation_states': [2, 3, 4]},
            'Pm': {'Z': 61, 'mass': 145.000, 'radius': 2.05, 'electronegativity': 1.13, 'ionization_energy': 5.582,
                   'electron_affinity': 0.500, 'oxidation_states': [3]},
            'Sm': {'Z': 62, 'mass': 150.360, 'radius': 2.04, 'electronegativity': 1.17, 'ionization_energy': 5.644,
                   'electron_affinity': 0.500, 'oxidation_states': [2, 3]},
            'Eu': {'Z': 63, 'mass': 151.964, 'radius': 2.03, 'electronegativity': 1.20, 'ionization_energy': 5.670,
                   'electron_affinity': 0.500, 'oxidation_states': [2, 3]},
            'Gd': {'Z': 64, 'mass': 157.250, 'radius': 2.01, 'electronegativity': 1.20, 'ionization_energy': 6.150,
                   'electron_affinity': 0.500, 'oxidation_states': [1, 2, 3]},
            'Tb': {'Z': 65, 'mass': 158.925, 'radius': 1.99, 'electronegativity': 1.22, 'ionization_energy': 5.864,
                   'electron_affinity': 0.500, 'oxidation_states': [1, 2, 3, 4]},
            'Dy': {'Z': 66, 'mass': 162.500, 'radius': 1.97, 'electronegativity': 1.23, 'ionization_energy': 5.939,
                   'electron_affinity': 0.500, 'oxidation_states': [2, 3, 4]},
            'Ho': {'Z': 67, 'mass': 164.930, 'radius': 1.96, 'electronegativity': 1.24, 'ionization_energy': 6.022,
                   'electron_affinity': 0.500, 'oxidation_states': [3]},
            'Er': {'Z': 68, 'mass': 167.259, 'radius': 1.94, 'electronegativity': 1.24, 'ionization_energy': 6.108,
                   'electron_affinity': 0.500, 'oxidation_states': [3]},
            'Tm': {'Z': 69, 'mass': 168.934, 'radius': 1.92, 'electronegativity': 1.25, 'ionization_energy': 6.184,
                   'electron_affinity': 0.500, 'oxidation_states': [2, 3]},
            'Yb': {'Z': 70, 'mass': 173.054, 'radius': 1.92, 'electronegativity': 1.10, 'ionization_energy': 6.254,
                   'electron_affinity': 0.500, 'oxidation_states': [2, 3]},
            'Lu': {'Z': 71, 'mass': 174.967, 'radius': 1.92, 'electronegativity': 1.27, 'ionization_energy': 5.426,
                   'electron_affinity': 0.500, 'oxidation_states': [3]},
            'Hf': {'Z': 72, 'mass': 178.490, 'radius': 1.87, 'electronegativity': 1.30, 'ionization_energy': 6.825,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 3, 4]},
            'Ta': {'Z': 73, 'mass': 180.948, 'radius': 1.70, 'electronegativity': 1.50, 'ionization_energy': 7.890,
                   'electron_affinity': 0.322, 'oxidation_states': [-1, 2, 3, 4, 5]},
            'W': {'Z': 74, 'mass': 183.840, 'radius': 1.62, 'electronegativity': 2.36, 'ionization_energy': 7.980,
                  'electron_affinity': 0.816, 'oxidation_states': [-2, -1, 0, 1, 2, 3, 4, 5, 6]},
            'Re': {'Z': 75, 'mass': 186.207, 'radius': 1.51, 'electronegativity': 1.90, 'ionization_energy': 7.880,
                   'electron_affinity': 0.150, 'oxidation_states': [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7]},
            'Os': {'Z': 76, 'mass': 190.230, 'radius': 1.44, 'electronegativity': 2.20, 'ionization_energy': 8.700,
                   'electron_affinity': 1.100, 'oxidation_states': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]},
            'Ir': {'Z': 77, 'mass': 192.217, 'radius': 1.41, 'electronegativity': 2.20, 'ionization_energy': 9.100,
                   'electron_affinity': 1.565, 'oxidation_states': [-3, -1, 0, 1, 2, 3, 4, 5, 6]},
            'Pt': {'Z': 78, 'mass': 195.084, 'radius': 1.36, 'electronegativity': 2.28, 'ionization_energy': 9.000,
                   'electron_affinity': 2.128, 'oxidation_states': [-2, 0, 1, 2, 3, 4, 5, 6]},
            'Au': {'Z': 79, 'mass': 196.967, 'radius': 1.36, 'electronegativity': 2.54, 'ionization_energy': 9.226,
                   'electron_affinity': 2.309, 'oxidation_states': [-1, 1, 2, 3, 5]},
            'Hg': {'Z': 80, 'mass': 200.590, 'radius': 1.32, 'electronegativity': 2.00, 'ionization_energy': 10.437,
                   'electron_affinity': 0.000, 'oxidation_states': [1, 2, 4]},
            'Tl': {'Z': 81, 'mass': 204.383, 'radius': 1.45, 'electronegativity': 1.62, 'ionization_energy': 6.108,
                   'electron_affinity': 0.200, 'oxidation_states': [1, 3]},
            'Pb': {'Z': 82, 'mass': 207.200, 'radius': 1.46, 'electronegativity': 2.33, 'ionization_energy': 7.417,
                   'electron_affinity': 0.364, 'oxidation_states': [-4, 2, 4]},
            'Bi': {'Z': 83, 'mass': 208.980, 'radius': 1.48, 'electronegativity': 2.02, 'ionization_energy': 7.289,
                   'electron_affinity': 0.946, 'oxidation_states': [-3, 3, 5]},
            'Po': {'Z': 84, 'mass': 209.000, 'radius': 1.40, 'electronegativity': 2.00, 'ionization_energy': 8.417,
                   'electron_affinity': 1.900, 'oxidation_states': [-2, 2, 4, 6]},
            'At': {'Z': 85, 'mass': 210.000, 'radius': 1.50, 'electronegativity': 2.20, 'ionization_energy': 9.500,
                   'electron_affinity': 2.800, 'oxidation_states': [-1, 1, 3, 5, 7]},
            'Rn': {'Z': 86, 'mass': 222.000, 'radius': 1.50, 'electronegativity': 2.20, 'ionization_energy': 10.745,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 6]},
            'Fr': {'Z': 87, 'mass': 223.000, 'radius': 2.60, 'electronegativity': 0.70, 'ionization_energy': 4.073,
                   'electron_affinity': 0.000, 'oxidation_states': [1]},
            'Ra': {'Z': 88, 'mass': 226.000, 'radius': 2.21, 'electronegativity': 0.90, 'ionization_energy': 5.279,
                   'electron_affinity': 0.000, 'oxidation_states': [2]},
            'Ac': {'Z': 89, 'mass': 227.000, 'radius': 2.15, 'electronegativity': 1.10, 'ionization_energy': 5.170,
                   'electron_affinity': 0.000, 'oxidation_states': [3]},
            'Th': {'Z': 90, 'mass': 232.038, 'radius': 2.06, 'electronegativity': 1.30, 'ionization_energy': 6.307,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 3, 4]},
            'Pa': {'Z': 91, 'mass': 231.036, 'radius': 2.00, 'electronegativity': 1.50, 'ionization_energy': 5.890,
                   'electron_affinity': 0.000, 'oxidation_states': [3, 4, 5]},
            'U': {'Z': 92, 'mass': 238.029, 'radius': 1.96, 'electronegativity': 1.38, 'ionization_energy': 6.194,
                  'electron_affinity': 0.000, 'oxidation_states': [3, 4, 5, 6]},
            'Np': {'Z': 93, 'mass': 237.000, 'radius': 1.90, 'electronegativity': 1.36, 'ionization_energy': 6.266,
                   'electron_affinity': 0.000, 'oxidation_states': [3, 4, 5, 6, 7]},
            'Pu': {'Z': 94, 'mass': 244.000, 'radius': 1.87, 'electronegativity': 1.28, 'ionization_energy': 6.026,
                   'electron_affinity': 0.000, 'oxidation_states': [3, 4, 5, 6, 7]},
            'Am': {'Z': 95, 'mass': 243.000, 'radius': 1.80, 'electronegativity': 1.30, 'ionization_energy': 5.974,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 3, 4, 5, 6]},
            'Cm': {'Z': 96, 'mass': 247.000, 'radius': 1.69, 'electronegativity': 1.30, 'ionization_energy': 5.991,
                   'electron_affinity': 0.000, 'oxidation_states': [3, 4]},
            'Bk': {'Z': 97, 'mass': 247.000, 'radius': 1.68, 'electronegativity': 1.30, 'ionization_energy': 6.198,
                   'electron_affinity': 0.000, 'oxidation_states': [3, 4]},
            'Cf': {'Z': 98, 'mass': 251.000, 'radius': 1.68, 'electronegativity': 1.30, 'ionization_energy': 6.282,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 3, 4]},
            'Es': {'Z': 99, 'mass': 252.000, 'radius': 1.65, 'electronegativity': 1.30, 'ionization_energy': 6.420,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 3]},
            'Fm': {'Z': 100, 'mass': 257.000, 'radius': 1.67, 'electronegativity': 1.30, 'ionization_energy': 6.500,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 3]},
            'Md': {'Z': 101, 'mass': 258.000, 'radius': 1.73, 'electronegativity': 1.30, 'ionization_energy': 6.580,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 3]},
            'No': {'Z': 102, 'mass': 259.000, 'radius': 1.76, 'electronegativity': 1.30, 'ionization_energy': 6.650,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 3]},
            'Lr': {'Z': 103, 'mass': 262.000, 'radius': 1.61, 'electronegativity': 1.30, 'ionization_energy': 4.900,
                   'electron_affinity': 0.000, 'oxidation_states': [3]},
            'Rf': {'Z': 104, 'mass': 267.000, 'radius': 1.57, 'electronegativity': 1.30, 'ionization_energy': 6.011,
                   'electron_affinity': 0.000, 'oxidation_states': [4]},
            'Db': {'Z': 105, 'mass': 268.000, 'radius': 1.49, 'electronegativity': 1.30, 'ionization_energy': 6.800,
                   'electron_affinity': 0.000, 'oxidation_states': [5]},
            'Sg': {'Z': 106, 'mass': 269.000, 'radius': 1.43, 'electronegativity': 1.30, 'ionization_energy': 7.800,
                   'electron_affinity': 0.000, 'oxidation_states': [6]},
            'Bh': {'Z': 107, 'mass': 270.000, 'radius': 1.41, 'electronegativity': 1.30, 'ionization_energy': 7.700,
                   'electron_affinity': 0.000, 'oxidation_states': [7]},
            'Hs': {'Z': 108, 'mass': 269.000, 'radius': 1.34, 'electronegativity': 1.30, 'ionization_energy': 7.600,
                   'electron_affinity': 0.000, 'oxidation_states': [8]},
            'Mt': {'Z': 109, 'mass': 278.000, 'radius': 1.29, 'electronegativity': 1.30, 'ionization_energy': 7.700,
                   'electron_affinity': 0.000, 'oxidation_states': [3, 4, 6]},
            'Ds': {'Z': 110, 'mass': 281.000, 'radius': 1.28, 'electronegativity': 1.30, 'ionization_energy': 7.800,
                   'electron_affinity': 0.000, 'oxidation_states': [6]},
            'Rg': {'Z': 111, 'mass': 282.000, 'radius': 1.21, 'electronegativity': 1.30, 'ionization_energy': 7.900,
                   'electron_affinity': 0.000, 'oxidation_states': [3]},
            'Cn': {'Z': 112, 'mass': 285.000, 'radius': 1.22, 'electronegativity': 1.30, 'ionization_energy': 8.000,
                   'electron_affinity': 0.000, 'oxidation_states': [2]},
            'Nh': {'Z': 113, 'mass': 286.000, 'radius': 1.36, 'electronegativity': 1.30, 'ionization_energy': 7.306,
                   'electron_affinity': 0.000, 'oxidation_states': [1, 3, 5]},
            'Fl': {'Z': 114, 'mass': 289.000, 'radius': 1.43, 'electronegativity': 1.30, 'ionization_energy': 8.539,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 4]},
            'Mc': {'Z': 115, 'mass': 290.000, 'radius': 1.62, 'electronegativity': 1.30, 'ionization_energy': 7.527,
                   'electron_affinity': 0.000, 'oxidation_states': [1, 3]},
            'Lv': {'Z': 116, 'mass': 293.000, 'radius': 1.75, 'electronegativity': 1.30, 'ionization_energy': 7.161,
                   'electron_affinity': 0.000, 'oxidation_states': [2, 4]},
            'Ts': {'Z': 117, 'mass': 294.000, 'radius': 1.65, 'electronegativity': 1.30, 'ionization_energy': 7.306,
                   'electron_affinity': 0.000, 'oxidation_states': [-1, 1, 3, 5]},
            'Og': {'Z': 118, 'mass': 294.000, 'radius': 1.57, 'electronegativity': 1.30, 'ionization_energy': 7.000,
                   'electron_affinity': 0.000, 'oxidation_states': [0, 2, 4]},
            # Additional properties for all elements
            'quantum_properties': {
                'spin_orbital_coupling': True,
                'magnetic_moment': True,
                'band_structure': True,
                'electron_configuration': True,
                'quantum_numbers': True
            },
            'crystal_properties': {
                'lattice_structure': True,
                'space_group': True,
                'lattice_parameters': True,
                'crystal_system': True
            },
            'material_specific_properties': {
                'thermal_conductivity': True,
                'electrical_conductivity': True,
                'mechanical_strength': True,
                'melting_point': True,
                'boiling_point': True
            }
        }
        # Add base properties for each element
        for symbol in self.element_symbols:  # Defined elsewhere in class
            elements[symbol] = {
                'atomic_number': self.atomic_numbers[symbol],
                'mass': self.atomic_masses[symbol],
                'radius': self.atomic_radii[symbol],
                'electronegativity': self.electronegativities[symbol],
                'ionization_energy': self.ionization_energies[symbol],
                'electron_affinity': self.electron_affinities[symbol],
                'oxidation_states': self.oxidation_states[symbol],
            }

        # Enhance with derived properties
        for symbol, props in elements.items():
            elements[symbol].update({
                'quantum_properties': {
                    'spin_orbital_coupling': self._calculate_soc(props),
                    'magnetic_moment': self._calculate_magnetic_moment(props),
                    'band_structure': self._calculate_band_structure(props),
                    'electron_configuration': self._get_electron_config(props),
                    'quantum_numbers': self._get_quantum_numbers(props)
                },
                'material_properties': {
                    'crystal_structure': self._predict_crystal_structure(props),
                    'coordination_number': self._predict_coordination(props),
                    'atomic_mobility': self._calculate_mobility(props),
                    'thermal_properties': self._get_thermal_properties(props),
                    'mechanical_properties': self._get_mechanical_properties(props)
                },
                'simulation_parameters': {
                    'dft_parameters': self._get_dft_params(props),
                    'molecular_dynamics': self._get_md_params(props),
                    'monte_carlo': self._get_mc_params(props)
                }
            })

            # Add application-specific parameters
            elements[symbol]['applications'] = {
                'photovoltaic': self._get_pv_suitability(props),
                'quantum_computing': self._get_quantum_suitability(props),
                'battery': self._get_battery_suitability(props),
                'catalyst': self._get_catalyst_suitability(props),
                'thermoelectric': self._get_thermoelectric_suitability(props),
                'superconductor': self._get_superconductor_suitability(props)
            }

        # Add metadata
        elements['metadata'] = {
            'timestamp': self._get_timestamp(),
            'version': self.VERSION,
            'parameter_sets': self._get_parameter_sets(),
            'validation_metrics': self._get_validation_metrics()
        }

        return elements

    def _load_property_ranges(self) -> Dict[str, Dict]:
        """Initialize all material property ranges from PDF"""
        return {
            'photovoltaic': {
                'bandgap': (0.5, 3.0),
                'efficiency': (15, 45),
                'carrier_mobility': (1, 1000),
                'absorption_coeff': (1e4, 1e6),
                'lifetime': (1e-9, 1e-6),
                'defect_density': (1e10, 1e16),
                'conductivity': (1e-6, 1e3),
                'thickness': (50, 1000)
            },
            'quantum_processor': {
                'coherence_time': (1e-6, 1e-3),
                'coupling_strength': (1e6, 1e9),
                'qubit_frequency': (1e9, 10e9),
                'anharmonicity': (-400e6, -100e6),
                'gate_fidelity': (0.99, 0.9999),
                'T1_relaxation': (1e-6, 1e-3),
                'T2_dephasing': (1e-6, 1e-3)
            },
            'biofuel_catalyst': {
                'surface_area': (100, 1000),
                'turnover_frequency': (0.1, 1000),
                'selectivity': (80, 99.9),
                'stability': (100, 10000),
                'conversion': (60, 99),
                'activation_energy': (20, 200),
                'reaction_rate': (0.01, 100)
            },
            'carbon_capture': {
                'co2_capacity': (0.5, 8),
                'binding_energy': (-80, -20),
                'selectivity_co2_n2': (10, 1000),
                'cycle_stability': (100, 10000),
                'working_capacity': (0.1, 5),
                'regeneration_energy': (20, 100),
                'kinetics': (0.1, 10)
            },
            'battery': {
                'capacity': (100, 500),
                'voltage': (1.5, 5),
                'cyclability': (500, 10000),
                'rate_capability': (1, 20),
                'energy_density': (200, 1000),
                'power_density': (100, 2000),
                'coulombic_efficiency': (95, 99.9)
            },
            'hydrogen_storage': {
                'gravimetric_capacity': (1, 10),
                'volumetric_density': (20, 70),
                'desorption_temp': (25, 150),
                'cycle_life': (100, 1000),
                'kinetics': (0.1, 10),
                'pressure_range': (1, 700),
                'enthalpy': (-90, -30)
            },
            'corrosion_resistant': {
                'corrosion_rate': (0.001, 1.0),
                'pitting_potential': (0.1, 1.0),
                'passivation_current': (1e-6, 1e-3),
                'repassivation_potential': (-0.5, 0.5),
                'stress_corrosion_resistance': (10, 100)
            },
            'smart_responsive': {
                'response_time': (0.1, 10),
                'actuation_strain': (1, 50),
                'cycle_durability': (100, 10000),
                'energy_efficiency': (20, 90),
                'sensitivity': (0.1, 10)
            },
            'lightweight_composite': {
                'density': (0.1, 5),
                'tensile_strength': (100, 3000),
                'youngs_modulus': (1, 500),
                'thermal_conductivity': (0.1, 500),
                'impact_strength': (10, 1000)
            },
            'transparent_conductive': {
                'conductivity': (100, 10000),
                'transparency': (80, 95),
                'work_function': (3, 6),
                'carrier_concentration': (1e19, 1e21),
                'mobility': (10, 100)
            },
            'biodegradable_polymer': {
                'degradation_rate': (0.1, 10),
                'tensile_strength': (10, 100),
                'crystallinity': (0, 90),
                'molecular_weight': (1000, 1000000),
                'glass_transition': (-50, 200)
            },
            'superconductor': {
                'critical_temperature': (4, 150),
                'critical_field': (0.1, 100),
                'critical_current': (1e4, 1e7),
                'coherence_length': (1, 100),
                'energy_gap': (0.1, 40)
            },
            'thermoelectric': {
                'seebeck_coefficient': (100, 1000),
                'electrical_conductivity': (100, 10000),
                'thermal_conductivity': (0.1, 10),
                'zT': (0.1, 3.0),
                'power_factor': (1e-4, 1e-2)
            },
            'magnetic_storage': {
                'coercivity': (100, 5000),
                'saturation_magnetization': (0.1, 2.5),
                'anisotropy_constant': (1e4, 1e7),
                'curie_temperature': (300, 1300),
                'domain_wall_width': (1, 100)
            }
        }

    def _generate_crystal_structure(self, material_type: str) -> tf.Tensor:
        """Generate realistic crystal structures with physical constraints"""
        # Select number of atoms based on material type
        num_atoms = tf.random.uniform([],
                                      self._get_structure_size_range(material_type)[0],
                                      self._get_structure_size_range(material_type)[1],
                                      dtype=tf.int32
                                      )

        # Generate lattice parameters
        a, b, c = tf.random.uniform([3], 3.0, 20.0)
        alpha, beta, gamma = tf.random.uniform([3], np.pi / 3, 2 * np.pi / 3)

        # Generate atomic positions
        positions = tf.random.uniform([num_atoms, 3])

        # Select elements based on material type
        atomic_numbers = self._select_elements_for_material(material_type, num_atoms)

        # Combine into structure tensor
        structure = tf.concat([
            positions,
            tf.cast(atomic_numbers[..., tf.newaxis], tf.float32),
            self._compute_local_environment(positions, atomic_numbers)
        ], axis=-1)

        return structure

    def _compute_graph_features(self,
                                structure: tf.Tensor,
                                material_type: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute comprehensive graph features for all network architectures"""
        # Extract positions and atomic numbers
        positions = structure[:, :3]
        atomic_numbers = tf.cast(structure[:, 3], tf.int32)

        # Compute distance matrix
        distances = tf.sqrt(tf.reduce_sum(
            (positions[:, None, :] - positions[None, :, :]) ** 2,
            axis=-1
        ))

        # Build adjacency matrix with distance threshold
        adjacency = tf.cast(distances < self.cutoff_radius, tf.float32)

        # Compute edge features
        edge_features = self._compute_edge_features(
            positions, atomic_numbers, distances, adjacency
        )

        # Compute node features
        node_features = self._compute_node_features(
            atomic_numbers, adjacency, material_type
        )

        return node_features, adjacency, edge_features

    def generate_batch(self,
                       material_type: str,
                       batch_size: Optional[int] = None) -> Dict[str, tf.Tensor]:
        """Generate a batch of data for training"""
        if batch_size is None:
            batch_size = self.batch_size

        structures = []
        node_features_list = []
        adjacency_list = []
        edge_features_list = []
        targets_list = []

        for _ in range(batch_size):
            # Generate base structure
            structure = self._generate_crystal_structure(material_type)

            # Compute graph features
            node_feats, adj, edge_feats = self._compute_graph_features(
                structure, material_type
            )

            # Generate target properties
            targets = self._generate_target_properties(material_type)

            structures.append(structure)
            node_features_list.append(node_feats)
            adjacency_list.append(adj)
            edge_features_list.append(edge_feats)
            targets_list.append(targets)

        # Combine into batch
        batch_data = {
            'structures': tf.stack(structures),
            'node_features': tf.stack(node_features_list),
            'adjacency': tf.stack(adjacency_list),
            'edge_features': tf.stack(edge_features_list),
            'targets': tf.stack(targets_list),
            'material_type': [material_type] * batch_size
        }

        return batch_data

    def create_dataset(self,
                       material_type: str,
                       num_samples: int,
                       shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset for training"""

        def generator():
            for _ in range(num_samples):
                yield self.generate_batch(material_type, 1)

        # Define output shapes and types
        output_signature = {
            'structures': tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            'node_features': tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            'adjacency': tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            'edge_features': tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            'targets': tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            'material_type': tf.TensorSpec(shape=(None,), dtype=tf.string)
        }

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )

        if shuffle:
            dataset = dataset.shuffle(1000)

        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def save_dataset(self,
                     material_type: str,
                     num_samples: int,
                     filepath: str):
        """Save generated dataset to file"""
        dataset = self.create_dataset(material_type, num_samples, shuffle=False)

        with h5py.File(filepath, 'w') as f:
            for i, batch in enumerate(dataset):
                group = f.create_group(f'batch_{i}')
                for key, value in batch.items():
                    if key != 'material_type':
                        group.create_dataset(key, data=value.numpy())
                    else:
                        group.create_dataset(
                            key,
                            data=[str(v.numpy()) for v in value]
                        )


if __name__ == "__main__":
    # Initialize generator
    generator = MaterialsDataGenerator(batch_size=32)

    # Generate datasets for all material types
    material_types = [
        'photovoltaic', 'quantum_processor', 'biofuel_catalyst',
        'carbon_capture', 'battery', 'hydrogen_storage',
        'thermoelectric', 'magnetic_storage', 'corrosion_resistant',
        'smart_responsive', 'lightweight_composite', 'transparent_conductive',
        'biodegradable_polymer', 'superconductor', 'catalytic_converter'
    ]

    for mat_type in material_types:
        print(f"Generating dataset for {mat_type}...")
        generator.save_dataset(
            mat_type,
            num_samples=100000,
            filepath=f'data/{mat_type}_dataset.h5'
        )