import tensorflow as tf
import numpy as np
import pandas as pd
import json
import h5py
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm, lognorm


@dataclass
class MaterialProperties:
    density: float
    elastic_modulus: float
    poisson_ratio: float
    thermal_expansion: float
    thermal_conductivity: float
    specific_heat: float
    glass_transition_temp: float
    melting_point: Optional[float]


class MultiFormatDataGenerator:
    def __init__(self, config: Dict = None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self._initialize_materials()
        self._setup_environmental_params()
        self.config = config or self._default_config()

    def _initialize_materials(self):
        self.materials = {
            'hdpe': MaterialProperties(
                density=0.97, elastic_modulus=1100.0, poisson_ratio=0.42,
                thermal_expansion=1.2e-4, thermal_conductivity=0.48,
                specific_heat=2300.0, glass_transition_temp=390.0,
                melting_point=410.0
            ),
            'pp': MaterialProperties(
                density=0.91, elastic_modulus=1500.0, poisson_ratio=0.45,
                thermal_expansion=1.5e-4, thermal_conductivity=0.22,
                specific_heat=1920.0, glass_transition_temp=280.0,
                melting_point=434.0
            ),
            'asphalt': MaterialProperties(
                density=2.36, elastic_modulus=3000.0, poisson_ratio=0.35,
                thermal_expansion=3.5e-5, thermal_conductivity=0.75,
                specific_heat=920.0, glass_transition_temp=323.0,
                melting_point=None
            )
        }

    def _setup_environmental_params(self):
        self.env_params = {
            'temp_range': (-20.0, 60.0),
            'rainfall': lognorm(s=1.0, scale=25.0),
            'traffic': lognorm(s=0.5, scale=80.0),
            'uv': norm(loc=5.8, scale=2.0)
        }

    @staticmethod
    def _default_config():
        return {
            'grid_size': 32,
            'porosity_range': (0.15, 0.30),
            'permeability_range': (1e-5, 1e-3),
            'batch_size': 64
        }

    def generate_sample(self) -> Tuple[Dict, Dict]:
        composition = self._generate_composition()
        conditions = self._generate_conditions()
        grid = self._generate_grid(composition)
        properties = self._calculate_properties(composition, conditions)

        return {
            'composition': composition,
            'conditions': conditions,
            'grid': grid.numpy()
        }, properties

    def save_multiple_formats(self, n_samples: int, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        samples = [self.generate_sample() for _ in range(n_samples)]

        # Save as CSV
        self._save_csv(samples, output_dir / 'data.csv')

        # Save as JSON
        self._save_json(samples, output_dir / 'data.json')

        # Save as HDF5
        self._save_hdf5(samples, output_dir / 'data.h5')

    def _save_csv(self, samples: List[Tuple], path: Path):
        """Save flat data structure to CSV"""
        rows = []
        for features, properties in samples:
            row = {
                'hdpe_content': features['composition']['hdpe'],
                'pp_content': features['composition']['pp'],
                'asphalt_content': features['composition']['asphalt'],
                'temperature': features['conditions']['temperature'],
                'rainfall': features['conditions']['rainfall'],
                'traffic_load': features['conditions']['traffic_load']
            }
            row.update(properties)
            rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False)

    def _save_json(self, samples: List[Tuple], path: Path):
        """Save hierarchical data structure to JSON"""
        data = {
            'metadata': {
                'date_generated': datetime.now().isoformat(),
                'n_samples': len(samples),
                'config': self.config
            },
            'samples': [
                {
                    'features': {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in features.items()
                    },
                    'properties': properties
                }
                for features, properties in samples
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_hdf5(self, samples: List[Tuple], path: Path):
        """Save numerical data optimized for HDF5"""
        with h5py.File(path, 'w') as f:
            # Create groups
            features = f.create_group('features')
            properties = f.create_group('properties')

            # Store numeric data as datasets
            for key in ['composition', 'conditions']:
                data = np.array([s[0][key] for s in samples])
                features.create_dataset(key, data=data)

            # Store grid structures
            grids = np.array([s[0]['grid'] for s in samples])
            features.create_dataset('grid', data=grids, compression='gzip')

            # Store properties
            prop_data = np.array([list(s[1].values()) for s in samples])
            properties.create_dataset('values', data=prop_data)
            properties.attrs['keys'] = list(samples[0][1].keys())

            # Store metadata
            f.attrs['date_generated'] = datetime.now().isoformat()
            f.attrs['n_samples'] = len(samples)

    def _generate_composition(self) -> Dict[str, float]:
        while True:
            hdpe = np.random.uniform(0.1, 0.4)
            pp = np.random.uniform(0.1, 0.4)
            if hdpe + pp <= 0.8:
                return {
                    'hdpe': hdpe,
                    'pp': pp,
                    'asphalt': 1.0 - (hdpe + pp)
                }

    def _generate_conditions(self) -> Dict[str, float]:
        return {
            'temperature': np.random.uniform(*self.env_params['temp_range']),
            'rainfall': float(self.env_params['rainfall'].rvs()),
            'traffic_load': float(self.env_params['traffic'].rvs())
        }

    @tf.function
    def _generate_grid(self, composition: Dict[str, float]) -> tf.Tensor:
        size = self.config['grid_size']
        grid = tf.zeros((size, size, size, 3))

        for i in range(0, size, 4):
            for j in range(0, size, 4):
                mask = self._create_grid_mask(i, j, size)
                materials = tf.constant([
                    composition['hdpe'],
                    composition['pp'],
                    composition['asphalt']
                ])
                grid += mask * materials

        return grid

    def _calculate_properties(self, composition: Dict[str, float],
                              conditions: Dict[str, float]) -> Dict[str, float]:
        # Implementation of material property calculations
        temp = conditions['temperature']

        def temp_adjusted_modulus(mat: MaterialProperties) -> float:
            return mat.elastic_modulus * np.exp(
                -0.05 * (temp - 293.15) / (mat.glass_transition_temp - temp + 1e-6)
            )

        moduli = {name: temp_adjusted_modulus(props)
                  for name, props in self.materials.items()}

        E_eff = sum(composition[mat] * E for mat, E in moduli.items())

        return {
            'elastic_modulus': E_eff,
            'porosity': self._calculate_porosity(composition),
            'permeability': self._calculate_permeability(composition),
            'thermal_conductivity': self._calculate_thermal_conductivity(composition)
        }


if __name__ == '__main__':
    generator = MultiFormatDataGenerator()
    generator.save_multiple_formats(
        n_samples=10000,
        output_dir='data/road_materials'
    )