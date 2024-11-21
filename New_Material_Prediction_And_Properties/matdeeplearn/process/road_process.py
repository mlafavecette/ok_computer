import tensorflow as tf
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import json


@dataclass
class ProcessingConfig:
    input_dir: Path = Path('/data/plastic_grid_road/raw')
    output_dir: Path = Path('/data/plastic_grid_road/processed')
    batch_size: int = 1000


class RoadMaterialProcessor:
    """Process raw material data for ML models."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.setup_dirs()
        self.logger = self._setup_logging()

    def setup_dirs(self):
        """Create directory structure."""
        dirs = [
            self.config.output_dir,
            self.config.output_dir / 'graphs',
            self.config.output_dir / 'features',
            self.config.output_dir / 'metadata'
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    @tf.function
    def process_batch(
            self,
            positions: tf.Tensor,
            numbers: tf.Tensor,
            cell: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Process batch of structures."""
        # Compute distances
        expand_pos = tf.expand_dims(positions, 1)
        dists = tf.norm(expand_pos - tf.expand_dims(positions, 2), axis=-1)

        # Create adjacency matrix
        adj = tf.cast(dists <= 5.0, tf.float32)

        # Get edge indices
        edge_index = tf.where(adj)
        edge_attr = tf.gather_nd(dists, edge_index)

        # Compute node features
        node_features = self._compute_node_features(positions, numbers)

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': tf.expand_dims(edge_attr, -1)
        }

    def _compute_node_features(
            self,
            positions: tf.Tensor,
            numbers: tf.Tensor
    ) -> tf.Tensor:
        """Compute node-level features."""
        # One-hot encode atomic numbers
        one_hot = tf.one_hot(numbers, depth=100)

        # Add positional encoding
        pos_encoding = self._positional_encoding(positions)

        return tf.concat([one_hot, pos_encoding], axis=-1)

    def _positional_encoding(self, positions: tf.Tensor) -> tf.Tensor:
        """Compute positional encodings."""
        d_model = 32
        pe = np.zeros((positions.shape[0], d_model))
        position = positions.numpy()

        for i in range(0, d_model, 2):
            pe[:, i] = np.sin(position[:, 0] * (1.0 / 10000 ** (i / d_model)))
            pe[:, i + 1] = np.cos(position[:, 0] * (1.0 / 10000 ** (i / d_model)))

        return tf.constant(pe, dtype=tf.float32)

    def process_dataset(self):
        """Process complete dataset."""
        input_files = list(self.config.input_dir.glob('*.h5'))

        for file_path in tqdm(input_files):
            try:
                self._process_file(file_path)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")

        self._save_metadata(len(input_files))

    def _process_file(self, file_path: Path):
        """Process single data file."""
        with h5py.File(file_path, 'r') as f_in:
            data = {key: f_in[key][()] for key in f_in.keys()}

        # Process in batches
        processed_data = []
        for i in range(0, len(data['positions']), self.config.batch_size):
            batch = {
                k: v[i:i + self.config.batch_size]
                for k, v in data.items()
            }
            processed_batch = self.process_batch(
                tf.constant(batch['positions']),
                tf.constant(batch['numbers']),
                tf.constant(batch['cell'])
            )
            processed_data.append(processed_batch)

        # Save processed data
        output_path = self.config.output_dir / 'processed' / file_path.name
        with h5py.File(output_path, 'w') as f_out:
            for key in processed_data[0].keys():
                combined = tf.concat([batch[key] for batch in processed_data], axis=0)
                f_out.create_dataset(key, data=combined.numpy())

    def _save_metadata(self, num_files: int):
        """Save processing metadata."""
        metadata = {
            'num_files': num_files,
            'config': vars(self.config),
            'feature_dims': {
                'node_features': 132,  # 100 + 32 positional
                'edge_features': 1
            }
        }

        with open(self.config.output_dir / 'metadata/processing_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    config = ProcessingConfig()
    processor = RoadMaterialProcessor(config)
    processor.process_dataset()