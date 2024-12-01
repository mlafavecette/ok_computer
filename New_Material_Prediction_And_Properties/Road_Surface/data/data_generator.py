import tensorflow as tf
import numpy as np
import h5py
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import logging


class MunicipalRoadDataGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_material_properties()
        self._init_environmental_conditions()
        self.start_date = datetime(2019, 1, 1)

    def _init_material_properties(self):
        self.materials = {
            'hdpe': {
                'density': 0.97,
                'elastic_modulus': np.random.normal(1100.0, 50.0, 1000),
                'thermal_exp': 1.2e-4,
                'degradation_rate': 0.001,
                'cost': 2.5
            },
            'pp': {
                'density': 0.91,
                'elastic_modulus': np.random.normal(1500.0, 75.0, 1000),
                'thermal_exp': 1.5e-4,
                'degradation_rate': 0.0015,
                'cost': 2.2
            },
            'asphalt': {
                'density': 2.36,
                'elastic_modulus': np.random.normal(3000.0, 150.0, 1000),
                'thermal_exp': 3.5e-5,
                'degradation_rate': 0.002,
                'porosity': np.random.uniform(0.15, 0.25, 1000),
                'cost': 1.8
            }
        }

    def _init_environmental_conditions(self):
        # Generate 5 years of daily weather data for multiple climate zones
        days = 365 * 5
        self.climate_data = {
            'temperate': {
                'temperature': self._generate_seasonal_temps(days, 15, 25),
                'rainfall': self._generate_rainfall(days, 800, 200),
                'freeze_thaw': self._generate_freeze_thaw(days, 20)
            },
            'continental': {
                'temperature': self._generate_seasonal_temps(days, 20, 35),
                'rainfall': self._generate_rainfall(days, 600, 150),
                'freeze_thaw': self._generate_freeze_thaw(days, 40)
            },
            'coastal': {
                'temperature': self._generate_seasonal_temps(days, 10, 15),
                'rainfall': self._generate_rainfall(days, 1200, 300),
                'freeze_thaw': self._generate_freeze_thaw(days, 5)
            }
        }

    def generate_dataset(self, output_path: str, samples: int = 100000):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path / 'municipal_data.h5', 'w') as hf:
            # Grid structures (500MB)
            grids = self._generate_grid_structures(samples)
            hf.create_dataset('grid_structures', data=grids, compression='gzip')

            # Material compositions (100MB)
            compositions = self._generate_compositions(samples)
            hf.create_dataset('compositions', data=compositions)

            # Load testing (200MB)
            load_tests = self._generate_load_tests(samples)
            hf.create_dataset('load_tests', data=load_tests, compression='gzip')

            # Environmental data (300MB)
            env_data = self._generate_environmental_data(samples)
            hf.create_dataset('environmental', data=env_data, compression='gzip')

            # Performance history (400MB)
            performance = self._generate_performance_history(samples)
            hf.create_dataset('performance', data=performance, compression='gzip')

        # Generate CSV files for easy inspection
        self._save_summary_csv(output_path, samples)

    def _generate_grid_structures(self, samples: int) -> np.ndarray:
        grids = np.zeros((samples, 32, 32, 32, 3))
        for i in range(samples):
            grid = self._create_single_grid()
            grids[i] = grid
        return grids

    def _create_single_grid(self) -> np.ndarray:
        grid = np.zeros((32, 32, 32, 3))
        # Complex grid pattern generation with variations
        spacing = np.random.uniform(3, 5)
        thickness = np.random.uniform(0.2, 0.4)

        for i in range(0, 32, int(spacing)):
            for j in range(0, 32, int(spacing)):
                grid[i:i + int(thickness), :, :, 0] = 1  # HDPE
                grid[:, j:j + int(thickness), :, 1] = 1  # PP

        grid[:, :, :, 2] = 1 - (grid[:, :, :, 0] + grid[:, :, :, 1])  # Asphalt
        return grid

    def _generate_load_tests(self, samples: int) -> np.ndarray:
        tests = []
        for _ in range(samples):
            cycles = 10000
            loads = np.random.normal(80, 20, cycles)  # kN
            deflections = np.random.normal(2, 0.5, cycles)  # mm
            recovery = np.random.normal(98, 1, cycles)  # percent
            tests.append(np.column_stack((loads, deflections, recovery)))
        return np.array(tests)

    def _generate_environmental_data(self, samples: int) -> np.ndarray:
        data = []
        for _ in range(samples):
            climate = np.random.choice(list(self.climate_data.keys()))
            years_data = {
                'daily_temp': self.climate_data[climate]['temperature'][:365 * 5],
                'rainfall': self.climate_data[climate]['rainfall'][:365 * 5],
                'freeze_thaw': self.climate_data[climate]['freeze_thaw'][:365 * 5]
            }
            data.append(np.column_stack(list(years_data.values())))
        return np.array(data)

    def _generate_performance_history(self, samples: int) -> np.ndarray:
        history = []
        for _ in range(samples):
            years = 5
            days = years * 365
            structural_integrity = self._generate_degradation(days, 0.999, 0.0001)
            water_drainage = self._generate_degradation(days, 0.995, 0.0002)
            surface_wear = self._generate_degradation(days, 0.998, 0.0001)
            history.append(np.column_stack((structural_integrity, water_drainage, surface_wear)))
        return np.array(history)

    def _generate_seasonal_temps(self, days: int, mean: float, amplitude: float) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi * days / 365, days)
        return mean + amplitude * np.sin(t) + np.random.normal(0, 2, days)

    def _generate_rainfall(self, days: int, annual_mean: float, std: float) -> np.ndarray:
        daily_mean = annual_mean / 365
        return np.random.gamma(2, daily_mean / 2, days)

    def _generate_freeze_thaw(self, days: int, annual_cycles: int) -> np.ndarray:
        cycles = np.zeros(days)
        cycle_days = np.random.choice(days, annual_cycles * 5, replace=False)
        cycles[cycle_days] = 1
        return cycles

    def _generate_degradation(self, days: int, initial: float, rate: float) -> np.ndarray:
        t = np.arange(days)
        return initial * np.exp(-rate * t) + np.random.normal(0, 0.001, days)

    def _generate_compositions(self, samples: int) -> np.ndarray:
        compositions = []
        for _ in range(samples):
            hdpe = np.random.uniform(0.1, 0.4)
            pp = np.random.uniform(0.1, 0.4)
            if hdpe + pp <= 0.8:
                asphalt = 1.0 - (hdpe + pp)
                compositions.append([hdpe, pp, asphalt])
        return np.array(compositions)

    def _save_summary_csv(self, output_path: Path, samples: int):
        summary_data = {
            'sample_id': range(samples),
            'date_generated': [self.start_date + timedelta(days=i % (5 * 365)) for i in range(samples)],
            'location': np.random.choice(list(self.climate_data.keys()), samples),
            'installation_cost': np.random.normal(150, 20, samples)
        }
        pd.DataFrame(summary_data).to_csv(output_path / 'summary.csv', index=False)


if __name__ == '__main__':
    generator = MunicipalRoadDataGenerator()
    generator.generate_dataset('data/municipal_approval', samples=100000)