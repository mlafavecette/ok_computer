import numpy as np
import h5py
from pathlib import Path
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import json
import os


@dataclass
class AsphaltProperties:
    permeability: float = 1e-5  # cm/s
    void_ratio: float = 0.04  # fraction
    thickness: float = 4.0  # inches
    age: float = 0.0  # years
    surface_condition: float = 1.0  # 0-1 scale


class AsphaltDrainageGenerator:  # This name must match what's used in __main__
    def __init__(self):
        # Get the current file's directory and create outputs folder there
        self.output_dir = Path("outputs")  # Creates 'outputs' in current directory
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'raw').mkdir(exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)

        self.rainfall_data = self._load_nyc_rainfall_patterns()
        self.infrastructure_zones = self._define_critical_zones()

    def _load_nyc_rainfall_patterns(self) -> Dict:
        """NYC rainfall patterns based on historical data."""
        return {
            'annual_rainfall': 46.23,  # inches
            'monthly_distribution': [
                3.1, 2.9, 4.2, 4.5, 4.2, 4.6,
                4.6, 4.1, 4.1, 3.8, 3.6, 3.3
            ],
            'intensity_patterns': {
                'light': {'rate': 0.1, 'probability': 0.5},
                'moderate': {'rate': 0.3, 'probability': 0.3},
                'heavy': {'rate': 1.0, 'probability': 0.15},
                'extreme': {'rate': 2.0, 'probability': 0.05}
            }
        }

    def _define_critical_zones(self) -> List[Dict]:
        """Define critical infrastructure zones."""
        return [
            {
                'name': 'subway_entrance',
                'water_sensitivity': 0.9,
                'drainage_capacity': 0.3,
                'flood_risk': 0.8
            },
            {
                'name': 'power_substation',
                'water_sensitivity': 0.85,
                'drainage_capacity': 0.4,
                'flood_risk': 0.75
            },
            {
                'name': 'tunnel_approach',
                'water_sensitivity': 0.95,
                'drainage_capacity': 0.25,
                'flood_risk': 0.9
            }
        ]

    def generate_sample(self) -> Dict:
        """Generate single drainage sample."""
        asphalt = AsphaltProperties(
            age=np.random.uniform(0, 15),
            surface_condition=np.random.uniform(0.5, 1.0)
        )

        rainfall_intensity = np.random.choice(
            list(self.rainfall_data['intensity_patterns'].keys()),
            p=[v['probability'] for v in self.rainfall_data['intensity_patterns'].values()]
        )

        zone = np.random.choice(self.infrastructure_zones)

        # Calculate drainage characteristics
        infiltration_rate = self._calculate_infiltration(asphalt, rainfall_intensity)
        runoff_coefficient = self._calculate_runoff(asphalt, zone)
        ponding_depth = self._calculate_ponding(infiltration_rate, rainfall_intensity)

        return {
            'asphalt_properties': {
                'permeability': asphalt.permeability,
                'void_ratio': asphalt.void_ratio,
                'thickness': asphalt.thickness,
                'age': asphalt.age,
                'surface_condition': asphalt.surface_condition
            },
            'rainfall': {
                'intensity': self.rainfall_data['intensity_patterns'][rainfall_intensity]['rate'],
                'duration': np.random.uniform(0.5, 6.0)  # hours
            },
            'infrastructure_zone': zone,
            'drainage_performance': {
                'infiltration_rate': infiltration_rate,
                'runoff_coefficient': runoff_coefficient,
                'ponding_depth': ponding_depth,
                'drainage_efficiency': self._calculate_efficiency(
                    infiltration_rate,
                    runoff_coefficient,
                    zone
                )
            }
        }

    def _calculate_infiltration(
            self,
            asphalt: AsphaltProperties,
            rainfall_intensity: str
    ) -> float:
        """Calculate water infiltration rate."""
        base_rate = asphalt.permeability
        age_factor = np.exp(-0.05 * asphalt.age)
        condition_factor = asphalt.surface_condition

        return base_rate * age_factor * condition_factor

    def _calculate_runoff(
            self,
            asphalt: AsphaltProperties,
            zone: Dict
    ) -> float:
        """Calculate runoff coefficient."""
        base_runoff = 0.95  # Typical for asphalt
        age_impact = 0.01 * asphalt.age
        slope_factor = np.random.uniform(0.02, 0.08)

        return min(1.0, base_runoff + age_impact + slope_factor)

    def _calculate_ponding(
            self,
            infiltration_rate: float,
            rainfall_intensity: str
    ) -> float:
        """Calculate water ponding depth."""
        rain_rate = self.rainfall_data['intensity_patterns'][rainfall_intensity]['rate']
        return max(0, (rain_rate - infiltration_rate) * 3600)  # mm

    def _calculate_efficiency(
            self,
            infiltration_rate: float,
            runoff_coefficient: float,
            zone: Dict
    ) -> float:
        """Calculate overall drainage efficiency."""
        base_efficiency = 1 - runoff_coefficient
        zone_factor = zone['drainage_capacity']

        return base_efficiency * zone_factor

    def generate_dataset(self, num_samples: int = 10000):
        """Generate complete dataset."""
        samples = []
        for _ in range(num_samples):
            samples.append(self.generate_sample())

        # Save raw data
        with h5py.File(self.output_dir / 'raw_data.h5', 'w') as f:
            for i, sample in enumerate(samples):
                grp = f.create_group(f'sample_{i}')
                self._save_to_hdf5(sample, grp)

        # Save processed data for ML models
        self._save_processed_data(samples)
        self._save_metadata(num_samples)

    def _save_to_hdf5(self, sample: Dict, grp: h5py.Group):
        """Save sample to HDF5 format."""
        for key, value in sample.items():
            if isinstance(value, dict):
                subgrp = grp.create_group(key)
                for k, v in value.items():
                    subgrp.create_dataset(k, data=v)
            else:
                grp.create_dataset(key, data=value)

    def _save_processed_data(self, samples: List[Dict]):
        """Save ML-ready data formats."""
        # Prepare data for VAE
        vae_data = np.array([
            [
                s['asphalt_properties']['permeability'],
                s['asphalt_properties']['void_ratio'],
                s['asphalt_properties']['age'],
                s['asphalt_properties']['surface_condition'],
                s['rainfall']['intensity'],
                s['rainfall']['duration'],
                s['drainage_performance']['infiltration_rate'],
                s['drainage_performance']['runoff_coefficient'],
                s['drainage_performance']['ponding_depth']
            ] for s in samples
        ])
        np.save(self.output_dir / 'vae_data.npy', vae_data)

        # Prepare data for CycleGAN
        cyclegan_data = {
            'asphalt': np.array([
                [
                    s['drainage_performance']['infiltration_rate'],
                    s['drainage_performance']['runoff_coefficient'],
                    s['drainage_performance']['ponding_depth']
                ] for s in samples
            ]),
            'target_properties': np.array([
                [
                    1 - s['drainage_performance']['runoff_coefficient'],
                    s['infrastructure_zone']['drainage_capacity'],
                    s['drainage_performance']['drainage_efficiency']
                ] for s in samples
            ])
        }
        np.save(self.output_dir / 'cyclegan_data.npy', cyclegan_data)

    def _save_metadata(self, num_samples: int):
        """Save dataset metadata."""
        metadata = {
            'num_samples': num_samples,
            'rainfall_patterns': self.rainfall_data,
            'infrastructure_zones': self.infrastructure_zones,
            'version': '1.0.0',
            'generation_date': pd.Timestamp.now().isoformat()
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    generator = AsphaltDrainageGenerator()  # This matches the class name above
    generator.generate_dataset(num_samples=10000)