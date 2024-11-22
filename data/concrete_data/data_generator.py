import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


class ConcreteDataGenerator:
    def __init__(self, output_dir: str = "concrete_training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Initialize network directories
        self.networks = {
            'schnet': self._init_network_dir('schnet'),
            'megnet': self._init_network_dir('megnet'),
            'mpnn': self._init_network_dir('mpnn'),
            'descriptor_nn': self._init_network_dir('descriptor_nn'),
            'cgcnn': self._init_network_dir('cgcnn'),
            'gcn': self._init_network_dir('gcn'),
            'gat_gnn': self._init_network_dir('gat_gnn'),
            'deep_gatgnn': self._init_network_dir('deep_gatgnn')
        }

        self.compositions = {
            'portland': {
                'components': {
                    'portland_cement': 0.15,
                    'aggregates': 0.75,
                    'water': 0.10
                },
                'properties': {
                    'compression_strength': 40.0,
                    'tensile_strength': 3.5,
                    'flexural_strength': 4.5,
                    'co2_emissions': 800.0,
                    'co2_capture': 0.0,
                    'mineralization': 0.0,
                    'direct_air_capture': 0.0
                }
            },
            'carbon_negative': {
                'components': {
                    'fly_ash': 0.30,
                    'ggbfs': 0.20,
                    'metakaolin': 0.05,
                    'silica_fume': 0.03,
                    'magnesium_silicate': 0.15,
                    'olivine': 0.15,
                    'biochar': 0.10,
                    'basalt_rock_dust': 0.02,
                    'calcium_carbonate': 0.03,
                    'magnesium_carbonate': 0.02,
                    'natural_aggregates': 0.10,
                    'recycled_aggregates': 0.05,
                    'sodium_hydroxide': 0.02,
                    'sodium_silicate': 0.04,
                    'silica_from_algae': 0.01,
                    'algae_biomass': 0.03,
                    'alginate_beads': 0.05,
                    'carbonation_accelerators': 0.02,
                    'water': 0.05
                },
                'properties': {
                    'compression_strength': 45.0,
                    'tensile_strength': 4.0,
                    'flexural_strength': 5.0,
                    'co2_emissions': 108.55,
                    'co2_capture': 435.0,
                    'mineralization': 155.0,
                    'direct_air_capture': 280.0
                }
            }
        }

    def _init_network_dir(self, name: str) -> Path:
        path = self.output_dir / name
        path.mkdir(exist_ok=True)
        return path

    def generate_base_data(self, n_samples: int) -> pd.DataFrame:
        data = []
        for composition_name, composition in self.compositions.items():
            for _ in range(n_samples):
                sample = {
                    'type': composition_name,
                    'temperature': np.random.uniform(15, 30),
                    'humidity': np.random.uniform(40, 90),
                    'curing_time': np.random.randint(7, 365)
                }

                # Component variations
                for component, base_value in composition['components'].items():
                    sample[f'comp_{component}'] = np.clip(
                        np.random.normal(base_value, base_value * 0.05),
                        0, 1
                    )

                # Property variations with correlations
                for prop, base_value in composition['properties'].items():
                    variation = np.random.normal(0, 0.1)
                    if prop in ['compression_strength', 'tensile_strength', 'flexural_strength']:
                        # Correlate strength properties
                        sample[prop] = max(0, base_value * (1 + variation))
                    elif 'co2' in prop:
                        # Create correlation between CO2 properties
                        sample[prop] = max(0, base_value * (1 + variation * 1.2))

                data.append(sample)

        return pd.DataFrame(data)

    def _generate_molecular_features(self, n_samples: int) -> np.ndarray:
        return np.random.normal(0, 1, (n_samples, 100, 3))

    def _generate_graph_features(self, n_samples: int) -> Dict[str, np.ndarray]:
        return {
            'nodes': np.random.normal(0, 1, (n_samples, 50, 32)),
            'edges': np.random.normal(0, 1, (n_samples, 50, 50)),
            'globals': np.random.normal(0, 1, (n_samples, 16))
        }

    def generate_network_data(self, n_samples: int = 10000) -> None:
        logger.info(f"Generating {n_samples} samples")
        base_data = self.generate_base_data(n_samples)

        for network, path in self.networks.items():
            logger.info(f"Processing {network}")

            if network in ['schnet']:
                molecular_data = {
                    'coordinates': self._generate_molecular_features(n_samples),
                    'properties': base_data
                }
                self._save_data(molecular_data, path / f'molecular_{self.timestamp}.json')

            elif network in ['megnet', 'mpnn', 'gcn', 'gat_gnn']:
                graph_data = {
                    'features': self._generate_graph_features(n_samples),
                    'properties': base_data
                }
                self._save_data(graph_data, path / f'graph_{self.timestamp}.json')

            elif network == 'descriptor_nn':
                self._save_data(base_data, path / f'properties_{self.timestamp}.csv')

            elif network in ['cgcnn', 'deep_gatgnn']:
                crystal_data = {
                    'lattice': np.random.normal(0, 1, (n_samples, 3, 3)),
                    'positions': self._generate_molecular_features(n_samples),
                    'properties': base_data
                }
                self._save_data(crystal_data, path / f'crystal_{self.timestamp}.json')

    def _save_data(self, data: Union[Dict, pd.DataFrame], path: Path) -> None:
        try:
            if path.suffix == '.csv':
                if isinstance(data, dict):
                    pd.DataFrame(data).to_csv(path, index=False)
                else:
                    data.to_csv(path, index=False)
            else:
                with open(path, 'w') as f:
                    json.dump(data, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved data to {path}")
        except Exception as e:
            logger.error(f"Error saving data to {path}: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        generator = ConcreteDataGenerator()
        generator.generate_network_data(n_samples=10000)
        logger.info("Data generation completed successfully")
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")