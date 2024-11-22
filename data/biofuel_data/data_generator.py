import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return json.JSONEncoder.default(self, obj)


class DataGenerator:
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        self.networks = {
            'schnet': self._init_network_dir('schnet'),
            'megnet': self._init_network_dir('megnet'),
            'mpnn': self._init_network_dir('mpnn'),
            'descriptor_nn': self._init_network_dir('descriptor_nn'),
            'cgcnn': self._init_network_dir('cgcnn'),
            'gcn': self._init_network_dir('gcn'),
            'gat_gnn': self._init_network_dir('gat_gnn'),
            'deep_gatgnn': self._init_network_dir('deep_gatgnn'),
            'super_schnet': self._init_network_dir('super_schnet'),
            'super_megnet': self._init_network_dir('super_megnet'),
            'super_mpnn': self._init_network_dir('super_mpnn'),
            'super_cgcnn': self._init_network_dir('super_cgcnn'),
            'quantum_gan': self._init_network_dir('quantum_gan')
        }

    def _init_network_dir(self, name: str) -> Path:
        path = self.output_dir / name
        path.mkdir(exist_ok=True)
        return path

    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        data = {}

        # Process parameters
        data['reactor_temperature'] = np.random.normal(550, 50, n_samples)
        data['pressure'] = np.random.uniform(3, 10, n_samples)
        data['microwave_power'] = np.random.uniform(80, 120, n_samples)
        data['steam_injection_rate'] = np.random.normal(450, 50, n_samples)
        data['catalyst_loading'] = np.random.uniform(0.1, 0.5, n_samples)

        # Plastic composition
        data['polyethylene_ratio'] = np.random.uniform(0, 1, n_samples)
        data['polypropylene_ratio'] = np.random.uniform(0, 1, n_samples)
        data['polystyrene_ratio'] = np.random.uniform(0, 1, n_samples)
        data['particle_size'] = np.random.normal(3, 0.5, n_samples)
        data['PVC_contaminant'] = np.random.uniform(0, 5, n_samples)

        # Fuel properties
        data['hydrocarbon_avg_chain_length'] = np.random.uniform(10, 25, n_samples)
        data['aromatic_content'] = np.random.uniform(5, 15, n_samples)
        data['sulfur_content'] = np.random.uniform(0, 50, n_samples)
        data['density'] = np.random.uniform(0.75, 0.85, n_samples)
        data['viscosity'] = np.random.uniform(1, 3, n_samples)

        # Times
        data['reaction_time'] = np.random.uniform(30, 60, n_samples)
        data['cooling_time'] = np.random.uniform(20, 40, n_samples)

        # Calculate target variables
        data['diesel_suitability_score'] = np.clip(
            0.8 * data['polypropylene_ratio'] + 0.1 * data['density'] - 0.05 * data['sulfur_content'], 0, 1)
        data['automobile_fuel_suitability_score'] = np.clip(
            0.5 * data['polyethylene_ratio'] + 0.2 * data['hydrocarbon_avg_chain_length'] - 0.05 * data['viscosity'], 0,
            1)
        data['jet_fuel_suitability_score'] = np.clip(
            0.3 * data['aromatic_content'] + 0.6 * data['density'] - 0.1 * data['PVC_contaminant'], 0, 1)

        data['diesel_yield'] = np.clip(data['diesel_suitability_score'] * 60, 10, 80)
        data['automobile_yield'] = np.clip(data['automobile_fuel_suitability_score'] * 40, 5, 60)
        data['jet_fuel_yield'] = np.clip(data['jet_fuel_suitability_score'] * 30, 0, 50)

        # Emissions and efficiency
        data['CO2_emissions'] = np.random.uniform(0.8, 1.2, n_samples) * 300
        data['VOC_emissions'] = np.random.uniform(0.9, 1.1, n_samples) * 50
        data['production_cost_per_liter'] = np.random.uniform(0.7, 1.5, n_samples) * data['automobile_yield']
        data['energy_efficiency'] = 200 / data['microwave_power']

        return pd.DataFrame(data)

    def _save_data(self, data: Union[Dict, pd.DataFrame], path: Path, format: str = 'csv') -> None:
        try:
            if format == 'csv':
                if isinstance(data, dict):
                    pd.DataFrame(data).to_csv(path, index=False)
                else:
                    data.to_csv(path, index=False)
            elif format == 'json':
                with open(path, 'w') as f:
                    json.dump(data, f, cls=NumpyEncoder, indent=2)
        except Exception as e:
            logger.error(f"Error saving data to {path}: {str(e)}")
            raise

    def _generate_atomic_features(self, n_samples: int) -> Dict[str, np.ndarray]:
        return {
            'atomic_numbers': np.random.randint(1, 119, (n_samples, 100)),
            'positions': np.random.normal(0, 1, (n_samples, 100, 3)),
            'charges': np.random.normal(0, 1, (n_samples, 100))
        }

    def _generate_graph_features(self, n_samples: int) -> Dict[str, np.ndarray]:
        n_nodes = 50
        return {
            'node_features': np.random.normal(0, 1, (n_samples, n_nodes, 32)),
            'edge_index': np.random.randint(0, n_nodes, (n_samples, 2, 100)),
            'edge_attr': np.random.normal(0, 1, (n_samples, 100, 16))
        }

    def generate_neural_network_data(self, n_samples: int = 10000) -> None:
        logger.info(f"Generating {n_samples} samples")
        try:
            base_data = self.generate_synthetic_data(n_samples)

            for network in self.networks:
                logger.info(f"Processing {network}")
                output_path = self.networks[network]

                atomic_features = self._generate_atomic_features(n_samples)
                graph_features = self._generate_graph_features(n_samples)

                if network in ['schnet', 'super_schnet']:
                    network_data = {
                        'atomic_features': atomic_features,
                        'properties': base_data.to_dict(orient='records')
                    }
                    self._save_data(network_data, output_path / f'{network}_data_{self.timestamp}.json', 'json')

                elif network in ['megnet', 'super_megnet', 'mpnn', 'super_mpnn']:
                    network_data = {
                        'graph_features': graph_features,
                        'properties': base_data.to_dict(orient='records')
                    }
                    self._save_data(network_data, output_path / f'{network}_data_{self.timestamp}.json', 'json')

                elif network in ['cgcnn', 'super_cgcnn']:
                    crystal_data = {
                        'lattice': np.random.normal(0, 1, (n_samples, 3, 3)),
                        'positions': atomic_features['positions'],
                        'properties': base_data.to_dict(orient='records')
                    }
                    self._save_data(crystal_data, output_path / f'{network}_data_{self.timestamp}.json', 'json')

                elif network in ['gcn', 'gat_gnn', 'deep_gatgnn']:
                    graph_data = {
                        'graph_features': graph_features,
                        'properties': base_data.to_dict(orient='records')
                    }
                    self._save_data(graph_data, output_path / f'{network}_data_{self.timestamp}.json', 'json')

                elif network == 'descriptor_nn':
                    self._save_data(base_data, output_path / f'descriptors_{self.timestamp}.csv')

                elif network == 'quantum_gan':
                    quantum_features = {
                        'quantum_states': np.random.normal(0, 1, (n_samples, 32)),
                        'amplitudes': np.random.normal(0, 1, (n_samples, 16)),
                        'properties': base_data.to_dict(orient='records')
                    }
                    self._save_data(quantum_features, output_path / f'quantum_data_{self.timestamp}.json', 'json')

                self._save_data(base_data, output_path / f'process_data_{self.timestamp}.csv')

        except Exception as e:
            logger.error(f"Error generating neural network data: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        generator = DataGenerator()
        generator.generate_neural_network_data(n_samples=10000)
        logger.info("Data generation completed successfully")
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")