import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scalers = {
            'node': StandardScaler(),
            'edge': StandardScaler(),
            'target': StandardScaler()
        }

    def load_data(self, data_path: str) -> Tuple[Dict, Dict, Dict]:
        data_path = Path(data_path)

        # Load different formats
        data = {
            'csv': pd.read_csv(data_path / 'data.csv'),
            'features': self._load_hdf5(data_path / 'data.h5', 'features'),
            'properties': self._load_hdf5(data_path / 'data.h5', 'properties')
        }

        return self._process_data(data)

    def _load_hdf5(self, path: str, group: str) -> Dict:
        with h5py.File(path, 'r') as f:
            return {k: f[group][k][:] for k in f[group].keys()}

    def _process_data(self, data: Dict) -> Tuple[Dict, Dict, Dict]:
        # Extract features
        node_features = self._extract_node_features(data)
        edge_features = self._extract_edge_features(data)
        targets = self._extract_targets(data)

        # Scale features
        scaled_node = self.scalers['node'].fit_transform(node_features)
        scaled_edge = self.scalers['edge'].fit_transform(edge_features)
        scaled_targets = self.scalers['target'].fit_transform(targets)

        # Split data
        train_idx = int(0.7 * len(scaled_node))
        val_idx = int(0.85 * len(scaled_node))

        splits = {}
        for name, start, end in [
            ('train', 0, train_idx),
            ('val', train_idx, val_idx),
            ('test', val_idx, None)
        ]:
            splits[name] = {
                'node_features': scaled_node[start:end],
                'edge_features': scaled_edge[start:end],
                'targets': scaled_targets[start:end]
            }

        return splits['train'], splits['val'], splits['test']

    def _extract_node_features(self, data: Dict) -> np.ndarray:
        return np.concatenate([
            data['features']['composition'],
            data['features']['grid'].reshape(len(data['features']['grid']), -1)
        ], axis=1)

    def _extract_edge_features(self, data: Dict) -> np.ndarray:
        return data['features']['conditions']

    def _extract_targets(self, data: Dict) -> np.ndarray:
        return data['properties']['values']

    def create_tf_dataset(self, features: Dict, batch_size: int) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(
            (
                (
                    features['node_features'],
                    features['edge_features']
                ),
                features['targets']
            )
        ).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)