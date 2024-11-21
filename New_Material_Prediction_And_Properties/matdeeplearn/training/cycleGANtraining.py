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
    permeability: float = 1e-5
    void_ratio: float = 0.04
    thickness: float = 4.0
    porosity: float = 0.15
    thermal_conductivity: float = 0.75
    compressive_strength: float = 2.4
    elastic_modulus: float = 3000


class NYCAsphaltGenerator:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = Path(current_dir) / 'outputs'
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'raw').mkdir(exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)

        self.rainfall_patterns = self._load_nyc_rainfall()
        self.zones = self._define_zones()

    def _load_nyc_rainfall(self) -> Dict:
        return {
            'annual_mean': 46.23,
            'monthly': {
                'winter': [3.1, 2.9, 4.2],
                'spring': [4.5, 4.2, 4.6],
                'summer': [4.6, 4.1, 4.1],
                'fall': [3.8, 3.6, 3.3]
            },
            'intensity': {
                'light': {'rate': 0.1, 'prob': 0.5},
                'moderate': {'rate': 0.3, 'prob': 0.3},
                'heavy': {'rate': 1.0, 'prob': 0.15},
                'extreme': {'rate': 2.0, 'prob': 0.05}
            }
        }

    def _define_zones(self) -> List[Dict]:
        return [
            {
                'type': 'subway_entrance',
                'flood_risk': 0.9,
                'traffic_load': 0.7,
                'drainage_req': 0.95
            },
            {
                'type': 'tunnel_approach',
                'flood_risk': 0.85,
                'traffic_load': 0.9,
                'drainage_req': 0.9
            },
            {
                'type': 'bridge_approach',
                'flood_risk': 0.8,
                'traffic_load': 0.85,
                'drainage_req': 0.85
            }
        ]

    def generate_sample(self, age: float) -> Dict:
        asphalt = AsphaltProperties()
        zone = np.random.choice(self.zones)
        rain_intensity = np.random.choice(
            list(self.rainfall_patterns['intensity'].keys()),
            p=[v['prob'] for v in self.rainfall_patterns['intensity'].values()]
        )

        # Calculate degradation
        age_factor = np.exp(-0.05 * age)
        traffic_impact = zone['traffic_load'] * (1 - age_factor)

        # Material properties with aging
        properties = {
            'permeability': asphalt.permeability * (1 + traffic_impact),
            'void_ratio': asphalt.void_ratio * (1 - 0.2 * traffic_impact),
            'thickness': asphalt.thickness * (1 - 0.1 * traffic_impact),
            'porosity': asphalt.porosity * (1 + 0.3 * traffic_impact),
            'thermal_conductivity': asphalt.thermal_conductivity * (1 - 0.15 * traffic_impact),
            'compressive_strength': asphalt.compressive_strength * (1 - 0.25 * traffic_impact),
            'elastic_modulus': asphalt.elastic_modulus * (1 - 0.2 * traffic_impact)
        }

        # Performance metrics
        rain_rate = self.rainfall_patterns['intensity'][rain_intensity]['rate']
        infiltration = properties['permeability'] * properties['porosity']
        runoff = max(0, rain_rate - infiltration)

        return {
            'material_properties': properties,
            'environmental_conditions': {
                'age': age,
                'rainfall_intensity': rain_rate,
                'temperature': np.random.normal(20, 5),
                'traffic_load': zone['traffic_load']
            },
            'performance_metrics': {
                'infiltration_rate': infiltration,
                'runoff_rate': runoff,
                'drainage_efficiency': 1 - (runoff / rain_rate) if rain_rate > 0 else 1,
                'structural_integrity': age_factor * (1 - traffic_impact),
                'surface_deterioration': traffic_impact
            },
            'zone_info': zone
        }

    def generate_dataset(self, num_samples: int = 10000):
        samples = []
        ages = np.random.uniform(0, 20, num_samples)

        for age in ages:
            samples.append(self.generate_sample(age))

        # Save raw data
        self._save_raw_data(samples)

        # Save ML-ready formats
        self._save_ml_data(samples)

        # Save metadata
        self._save_metadata(num_samples)

    def _save_raw_data(self, samples: List[Dict]):
        with h5py.File(self.output_dir / 'raw/data.h5', 'w') as f:
            for i, sample in enumerate(samples):
                grp = f.create_group(f'sample_{i}')
                for category, data in sample.items():
                    subgrp = grp.create_group(category)
                    for key, value in data.items():
                        subgrp.create_dataset(key, data=value)

    def _save_ml_data(self, samples: List[Dict]):
        def extract_features(sample):
            return np.concatenate([
                list(sample['material_properties'].values()),
                list(sample['environmental_conditions'].values()),
                [sample['zone_info']['flood_risk']],
                [sample['zone_info']['drainage_req']]
            ])

        features = np.array([extract_features(s) for s in samples])
        performance = np.array([
            list(s['performance_metrics'].values())
            for s in samples
        ])

        # Save for different model types
        np.save(self.output_dir / 'processed/vae_features.npy', features)
        np.save(self.output_dir / 'processed/vae_performance.npy', performance)

        np.save(self.output_dir / 'processed/cyclegan_data.npy', {
            'features': features,
            'performance': performance
        })

        with h5py.File(self.output_dir / 'processed/ml_ready.h5', 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('performance', data=performance)
            f.create_dataset(
                'zone_encodings',
                data=np.array([
                    [s['zone_info']['flood_risk'],
                     s['zone_info']['traffic_load'],
                     s['zone_info']['drainage_req']]
                    for s in samples
                ])
            )

    def _save_metadata(self, num_samples: int):
        metadata = {
            'num_samples': num_samples,
            'rainfall_patterns': self.rainfall_patterns,
            'zones': self.zones,
            'feature_dimensions': {
                'material_properties': 7,
                'environmental_conditions': 4,
                'performance_metrics': 5,
                'zone_info': 3
            },
            'generation_date': pd.Timestamp.now().isoformat()
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    generator = NYCAsphaltGenerator()
    generator.generate_dataset()
