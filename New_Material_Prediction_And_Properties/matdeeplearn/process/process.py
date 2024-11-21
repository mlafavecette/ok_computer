import os
import sys
import json
import warnings
import numpy as np
import ase.io
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial


@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline."""
    max_neighbors: int = 12
    max_radius: float = 8.0
    edge_features: bool = True
    node_features: bool = True
    global_features: bool = True
    use_voronoi: bool = False
    use_soap: bool = False
    use_sm: bool = False
    gaussian_distance_width: float = 0.2
    gaussian_distance_bins: int = 50
    batch_size: int = 32
    cache_data: bool = True
    num_parallel_calls: int = 8


class DataProcessor:
    """High-performance data processing for materials science data.

    This class handles the core data processing pipeline with optimizations
    for large-scale materials datasets.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._initialize_feature_generators()

    def _initialize_feature_generators(self):
        """Initialize feature generation components."""
        self.atom_featurizer = AtomFeaturizer()
        self.edge_featurizer = EdgeFeaturizer(
            max_neighbors=self.config.max_neighbors,
            max_radius=self.config.max_radius,
            num_bins=self.config.gaussian_distance_bins,
            width=self.config.gaussian_distance_width
        )

        if self.config.use_soap:
            self.soap_featurizer = SOAPFeaturizer()
        if self.config.use_sm:
            self.sm_featurizer = SMFeaturizer()

    @tf.function
    def process_structure(self,
                          crystal: ase.Atoms,
                          target: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        """Process single crystal structure with TF acceleration.

        Args:
            crystal: ASE atoms object
            target: Optional target property

        Returns:
            Dictionary of processed features
        """
        # Get basic structure information
        atomic_numbers = tf.convert_to_tensor(crystal.get_atomic_numbers())
        positions = tf.convert_to_tensor(crystal.get_positions())
        cell = tf.convert_to_tensor(crystal.get_cell())

        # Compute distance matrix efficiently
        distances = self._compute_distance_matrix(positions, cell)

        # Generate edge features
        edge_index, edge_features = self.edge_featurizer(distances)

        # Generate node features
        node_features = self.atom_featurizer(atomic_numbers)

        # Generate global features if needed
        if self.config.global_features:
            global_features = create_global_features(atomic_numbers)
        else:
            global_features = None

        # Compute additional descriptors if requested
        descriptors = {}
        if self.config.use_soap:
            descriptors['soap'] = self.soap_featurizer(crystal)
        if self.config.use_sm:
            descriptors['sm'] = self.sm_featurizer(crystal)

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'global_features': global_features,
            'descriptors': descriptors,
            'target': target
        }

    @tf.function
    def _compute_distance_matrix(self,
                                 positions: tf.Tensor,
                                 cell: tf.Tensor,
                                 use_pbc: bool = True) -> tf.Tensor:
        """Efficiently compute distance matrix with periodic boundary conditions.

        Args:
            positions: Atomic positions [num_atoms, 3]
            cell: Unit cell matrix [3, 3]
            use_pbc: Whether to use periodic boundary conditions

        Returns:
            Distance matrix [num_atoms, num_atoms]
        """
        diff = tf.expand_dims(positions, 1) - tf.expand_dims(positions, 0)

        if use_pbc:
            # Apply minimum image convention
            diff = diff - tf.round(diff @ tf.linalg.inv(cell)) @ cell

        return tf.norm(diff, axis=-1)

    def process_dataset(self,
                        structures: List[ase.Atoms],
                        targets: Optional[np.ndarray] = None,
                        num_parallel: Optional[int] = None) -> tf.data.Dataset:
        """Process multiple structures in parallel.

        Args:
            structures: List of crystal structures
            targets: Optional target properties
            num_parallel: Number of parallel processes

        Returns:
            TensorFlow dataset
        """
        num_parallel = num_parallel or self.config.num_parallel_calls

        # Create processing function
        def process_fn(structure, target=None):
            return self.process_structure(structure, target)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((structures, targets))

        # Apply parallel processing
        dataset = dataset.map(
            process_fn,
            num_parallel_calls=num_parallel
        )

        # Add batching and prefetching
        if self.config.batch_size > 0:
            dataset = dataset.batch(self.config.batch_size)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Add caching if requested
        if self.config.cache_data:
            dataset = dataset.cache()

        return dataset


def split_dataset(dataset: tf.data.Dataset,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  seed: Optional[int] = None) -> Tuple[tf.data.Dataset, ...]:
    """Split dataset into train/validation/test sets.

    Args:
        dataset: Input dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if seed is not None:
        tf.random.set_seed(seed)

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)

    train_dataset = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)

    return train_dataset, val_dataset, test_dataset


def split_dataset_cv(dataset: tf.data.Dataset,
                     num_folds: int = 5,
                     seed: Optional[int] = None) -> List[tf.data.Dataset]:
    """Create cross-validation folds.

    Args:
        dataset: Input dataset
        num_folds: Number of CV folds
        seed: Random seed

    Returns:
        List of dataset folds
    """
    if seed is not None:
        tf.random.set_seed(seed)

    dataset_size = len(dataset)
    fold_size = dataset_size // num_folds

    return [
        dataset.skip(i * fold_size).take(fold_size)
        for i in range(num_folds)
    ]


def create_global_features(atomic_numbers: tf.Tensor) -> tf.Tensor:
    """Create global features from atomic numbers.

    Args:
        atomic_numbers: Atomic numbers [num_atoms]

    Returns:
        Global features [num_features]
    """
    # Create composition features
    unique_elements, counts = tf.unique_with_counts(atomic_numbers)
    composition = tf.zeros(108)  # Max atomic number
    composition = tf.tensor_scatter_nd_update(
        composition,
        tf.expand_dims(unique_elements, 1),
        tf.cast(counts, tf.float32) / tf.cast(tf.size(atomic_numbers), tf.float32)
    )

    return composition


# Feature generation classes
class AtomFeaturizer(tf.keras.layers.Layer):
    """Generate atomic features."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_element_data()

    def _load_element_data(self):
        """Load elemental property data."""
        # Load from JSON file
        element_data_path = Path(__file__).parent / "element_data.json"
        with element_data_path.open() as f:
            self.element_data = json.load(f)

    def call(self, atomic_numbers: tf.Tensor) -> tf.Tensor:
        """Generate atomic features.

        Args:
            atomic_numbers: Atomic numbers [num_atoms]

        Returns:
            Atomic features [num_atoms, num_features]
        """
        features = []
        for z in atomic_numbers:
            element_features = [
                self.element_data[str(z.numpy())]["radius"],
                self.element_data[str(z.numpy())]["electronegativity"],
                self.element_data[str(z.numpy())]["ionization_energy"],
                self.element_data[str(z.numpy())]["electron_affinity"]
            ]
            features.append(element_features)

        return tf.convert_to_tensor(features, dtype=tf.float32)


class EdgeFeaturizer(tf.keras.layers.Layer):
    """Generate edge features."""

    def __init__(self,
                 max_neighbors: int = 12,
                 max_radius: float = 8.0,
                 num_bins: int = 50,
                 width: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.gaussian_distance = GaussianDistance(
            num_bins=num_bins,
            width=width
        )

    def call(self, distance_matrix: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate edge features from distance matrix.

        Args:
            distance_matrix: Distance matrix [num_atoms, num_atoms]

        Returns:
            Tuple of (edge_index, edge_features)
        """
        # Get edges within radius
        edges = tf.where(distance_matrix <= self.max_radius)

        # Sort edges by distance for each central atom
        distances = tf.gather_nd(distance_matrix, edges)
        sorted_indices = tf.argsort(distances)
        edges = tf.gather(edges, sorted_indices)
        distances = tf.gather(distances, sorted_indices)

        # Keep only max_neighbors edges per atom
        edge_index = []
        edge_features = []
        for i in range(tf.shape(distance_matrix)[0]):
            mask = edges[:, 0] == i
            node_edges = tf.boolean_mask(edges, mask)[:self.max_neighbors]
            node_distances = tf.boolean_mask(distances, mask)[:self.max_neighbors]

            edge_index.append(node_edges)
            edge_features.append(self.gaussian_distance(node_distances))

        edge_index = tf.concat(edge_index, axis=0)
        edge_features = tf.concat(edge_features, axis=0)

        return edge_index, edge_features


class GaussianDistance(tf.keras.layers.Layer):
    """Gaussian distance featurization."""

    def __init__(self,
                 num_bins: int = 50,
                 width: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.width = width

        # Initialize centers
        self.centers = tf.linspace(0.0, 1.0, num_bins)

    def call(self, distances: tf.Tensor) -> tf.Tensor:
        """Convert distances to Gaussian features.

        Args:
            distances: Distance values [num_edges]

        Returns:
            Gaussian features [num_edges, num_bins]
        """
        # Expand dimensions for broadcasting
        distances = tf.expand_dims(distances, -1)
        centers = tf.reshape(self.centers, (1, -1))

        # Compute Gaussian features
        return tf.exp(-0.5 * ((distances - centers) / self.width) ** 2)


# Optional feature classes (loaded only if dependencies available)
try:
    from dscribe.descriptors import SOAP, SineMatrix


    class SOAPFeaturizer(tf.keras.layers.Layer):
        """SOAP feature generation."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.soap = SOAP(
                species=['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl'],
                rcut=6.0,
                nmax=8,
                lmax=6
            )

        def call(self, structure: ase.Atoms) -> tf.Tensor:
            """Generate SOAP features."""
            return tf.convert_to_tensor(
                self.soap.create(structure),
                dtype=tf.float32
            )

    class SMFeaturizer(tf.keras.layers.Layer):
        """Sine Matrix feature generation for periodic systems."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, structure: ase.Atoms) -> tf.Tensor:
            """Generate Sine Matrix features.

            Args:
                structure: ASE atoms object

            Returns:
                SM features [num_features]
            """
            # Determine periodicity
            is_periodic = any(structure.pbc)

            # Create appropriate descriptor
            if is_periodic:
                descriptor = SineMatrix(
                    n_atoms_max=len(structure),
                    permutation="eigenspectrum",
                    sparse=False,
                    flatten=True
                )
            else:
                descriptor = CoulombMatrix(
                    n_atoms_max=len(structure),
                    permutation="eigenspectrum",
                    sparse=False,
                    flatten=True
                )

            features = descriptor.create(structure)
            return tf.convert_to_tensor(features, dtype=tf.float32)

except ImportError:
    warnings.warn("dscribe not found. SOAP and SM features disabled.")

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import VoronoiNN


    class VoronoiFeaturizer(tf.keras.layers.Layer):
        """Voronoi-based feature generation."""

        def __init__(self,
                     max_neighbors: int = 12,
                     min_weight: float = 0.01,
                     **kwargs):
            super().__init__(**kwargs)
            self.max_neighbors = max_neighbors
            self.min_weight = min_weight
            self.voronoi = VoronoiNN(
                cutoff=10.0,
                min_dist=0.1,
                allow_pathological=True
            )

        def call(self, structure: ase.Atoms) -> Tuple[tf.Tensor, tf.Tensor]:
            """Generate Voronoi-based connectivity and features.

            Args:
                structure: ASE atoms object

            Returns:
                Tuple of (edge_index, edge_features)
            """
            # Convert to pymatgen structure
            pmg_structure = Structure(
                lattice=structure.cell,
                species=structure.get_chemical_symbols(),
                coords=structure.get_positions(),
                coords_are_cartesian=True
            )

            edge_index = []
            edge_features = []

            # Get Voronoi neighbors for each atom
            for i in range(len(structure)):
                # Get neighbors and weights
                neighbors = self.voronoi.get_nn_info(pmg_structure, i)

                # Sort by weight and filter
                neighbors = sorted(neighbors,
                                   key=lambda x: x['weight'],
                                   reverse=True)
                neighbors = [n for n in neighbors
                             if n['weight'] >= self.min_weight][:self.max_neighbors]

                for neighbor in neighbors:
                    edge_index.append([i, neighbor['site_index']])

                    features = [
                        neighbor['weight'],  # Voronoi face area
                        neighbor['solid_angle'],  # Solid angle
                        neighbor['volume']  # Voronoi cell volume
                    ]
                    edge_features.append(features)

            return (tf.convert_to_tensor(edge_index, dtype=tf.int64),
                    tf.convert_to_tensor(edge_features, dtype=tf.float32))

except ImportError:
    warnings.warn("pymatgen not found. Voronoi features disabled.")


class MaterialsDataset:
    """Enhanced dataset class for materials data.

    This class provides efficient data loading and processing with
    sophisticated caching and prefetching strategies.
    """

    def __init__(self,
                 structures: List[ase.Atoms],
                 targets: Optional[np.ndarray] = None,
                 config: Optional[ProcessingConfig] = None,
                 cache_dir: Optional[str] = None):
        self.structures = structures
        self.targets = targets
        self.config = config or ProcessingConfig()
        self.cache_dir = cache_dir

        self.processor = DataProcessor(self.config)
        self._initialize_dataset()

    def _initialize_dataset(self):
        """Initialize TensorFlow dataset with caching."""
        if self.cache_dir and os.path.exists(self._cache_path):
            # Load from cache
            self.dataset = tf.data.Dataset.load(self._cache_path)
        else:
            # Process structures
            self.dataset = self.processor.process_dataset(
                self.structures,
                self.targets
            )

            # Save cache if requested
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                tf.data.Dataset.save(self.dataset, self._cache_path)

    @property
    def _cache_path(self) -> str:
        """Get cache file path."""
        return os.path.join(self.cache_dir, 'processed_dataset')

    def get_batch_iterator(self,
                           batch_size: int,
                           shuffle: bool = True,
                           seed: Optional[int] = None) -> tf.data.Dataset:
        """Get iterator for batched data.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            seed: Random seed

        Returns:
            Batched dataset iterator
        """
        dataset = self.dataset

        if shuffle:
            if seed is not None:
                tf.random.set_seed(seed)
            dataset = dataset.shuffle(buffer_size=len(self.structures))

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def get_cv_folds(self,
                     num_folds: int = 5,
                     seed: Optional[int] = None) -> List[tf.data.Dataset]:
        """Get cross-validation folds.

        Args:
            num_folds: Number of CV folds
            seed: Random seed

        Returns:
            List of dataset folds
        """
        return split_dataset_cv(self.dataset, num_folds, seed)

    def normalize_features(self,
                           feature_name: str,
                           method: str = 'standard') -> None:
        """Normalize features in-place.

        Args:
            feature_name: Name of feature to normalize
            method: Normalization method ('standard' or 'minmax')
        """
        features = []
        for data in self.dataset:
            if feature_name in data:
                features.append(data[feature_name])

        features = tf.concat(features, axis=0)

        if method == 'standard':
            mean = tf.reduce_mean(features, axis=0)
            std = tf.math.reduce_std(features, axis=0)
            normalizer = lambda x: (x - mean) / (std + 1e-8)
        elif method == 'minmax':
            min_val = tf.reduce_min(features, axis=0)
            max_val = tf.reduce_max(features, axis=0)
            normalizer = lambda x: (x - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Apply normalization
        def normalize_fn(data):
            if feature_name in data:
                data[feature_name] = normalizer(data[feature_name])
            return data

        self.dataset = self.dataset.map(normalize_fn)


class AsyncDataProcessor:
    """Asynchronous data processing for large datasets.

    This class implements efficient parallel processing strategies
    for handling large materials datasets.
    """

    def __init__(self,
                 num_workers: int = 4,
                 chunk_size: int = 1000):
        self.num_workers = num_workers
        self.chunk_size = chunk_size

    def process_structures(self,
                           structures: List[ase.Atoms],
                           processor: DataProcessor) -> tf.data.Dataset:
        """Process structures in parallel.

        Args:
            structures: List of crystal structures
            processor: DataProcessor instance

        Returns:
            Processed dataset
        """
        # Split into chunks
        chunks = [
            structures[i:i + self.chunk_size]
            for i in range(0, len(structures), self.chunk_size)
        ]

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            chunk_datasets = list(executor.map(
                partial(processor.process_dataset),
                chunks
            ))

        # Combine chunks
        return tf.data.Dataset.concatenate(*chunk_datasets)
