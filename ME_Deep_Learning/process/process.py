"""
Core data processing functionality for materials GNN models.

Implements sophisticated data processing with support for:
- Crystal structure handling
- Graph construction
- Feature engineering
- Chemical descriptors
- Efficient data loading
- Memory management

Features:
- ASE integration
- Voronoi analysis
- SOAP descriptors
- Coulomb matrices
- Periodic boundary conditions
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import numpy as np
import tensorflow as tf
import ase
import ase.io
from scipy import stats, interpolate
from sklearn.preprocessing import StandardScaler

from ME_Deep_Learning.utils.graph import GraphBatch, GraphData
from ME_Deep_Learning.utils.chemistry import ElementFeatures


class MaterialsDataset:
    """
    Dataset class for materials data handling.

    Features:
    - Efficient data loading
    - Memory management
    - Graph construction
    - Feature computation

    Args:
        data_path: Path to raw data
        processed_path: Path for processed data
        processing_args: Processing configuration
    """

    def __init__(
            self,
            data_path: str,
            processed_path: str = "processed",
            processing_args: Optional[Dict[str, Any]] = None
    ):
        self.data_path = Path(data_path)
        self.processed_path = self.data_path / processed_path
        self.processing_args = processing_args or {}

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize processing
        self.processed = False
        self.graphs: List[GraphData] = []
        self.targets: Dict[str, np.ndarray] = {}

    def process(self):
        """Process raw data into graph format."""
        self.logger.info(f"Processing data to {self.processed_path}")

        # Load atom features dictionary
        self.atom_features = self._load_atom_features()

        # Load target properties
        self.targets = self._load_targets()

        # Process structures
        self._process_structures()

        # Save processed data
        self._save_processed_data()

        self.processed = True

    def _load_atom_features(self) -> Dict[int, np.ndarray]:
        """Load atomic feature dictionary."""
        dict_source = self.processing_args.get("dictionary_source", "default")

        if dict_source == "default":
            self.logger.info("Using default element features")
            return ElementFeatures.get_default_features()
        elif dict_source == "blank":
            self.logger.warning("Using blank element features")
            return ElementFeatures.get_blank_features()
        else:
            dict_path = self.data_path / self.processing_args["dictionary_path"]
            self.logger.info(f"Loading element features from {dict_path}")
            with open(dict_path) as f:
                return json.load(f)

    def _load_targets(self) -> Dict[str, np.ndarray]:
        """Load target properties."""
        target_path = self.data_path / self.processing_args["target_path"]
        targets = {}

        with open(target_path) as f:
            header = next(f).strip().split(',')
            data = np.loadtxt(f, delimiter=',', dtype=str)

        structure_ids = data[:, 0]
        values = data[:, 1:].astype(np.float32)

        for i, name in enumerate(header[1:]):
            targets[name] = values[:, i]

        return targets

    def _process_structures(self):
        """Process crystal structures into graphs."""
        structures = self._load_structures()

        for i, (structure_id, structure) in enumerate(structures.items()):
            # Create graph data
            graph = self._structure_to_graph(structure_id, structure)

            # Add node features
            graph.node_features = self._compute_node_features(structure)

            # Add edge features if requested
            if self.processing_args.get("edge_features", False):
                graph.edge_features = self._compute_edge_features(structure, graph)

            # Add global features
            graph.global_features = self._compute_global_features(structure)

            # Add structure targets
            graph.targets = {k: v[i] for k, v in self.targets.items()}

            self.graphs.append(graph)

            if (i + 1) % 500 == 0:
                self.logger.info(f"Processed {i + 1} structures")

    def _load_structures(self) -> Dict[str, ase.Atoms]:
        """Load crystal structures."""
        data_format = self.processing_args.get("data_format", "cif")
        structures = {}

        if data_format == "db":
            # Load from ASE database
            db = ase.db.connect(self.data_path / "data.db")
            for row in db.select():
                structures[str(row.id)] = row.toatoms()
        else:
            # Load individual structure files
            for target_id in self.targets:
                structure = ase.io.read(
                    self.data_path / f"{target_id}.{data_format}"
                )
                structures[target_id] = structure

        return structures

    def _structure_to_graph(
            self,
            structure_id: str,
            structure: ase.Atoms
    ) -> GraphData:
        """Convert crystal structure to graph representation."""
        # Get distance matrix
        distances = structure.get_all_distances(mic=True)

        # Build graph based on distance cutoff and max neighbors
        edges, edge_weights = self._build_graph(
            distances,
            max_radius=self.processing_args["graph_max_radius"],
            max_neighbors=self.processing_args["graph_max_neighbors"]
        )

        # Create graph data object
        graph = GraphData(
            num_nodes=len(structure),
            edges=edges,
            edge_weights=edge_weights
        )

        graph.structure_id = structure_id

        return graph

    def _build_graph(
            self,
            distances: np.ndarray,
            max_radius: float,
            max_neighbors: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build graph from distance matrix."""
        # Apply distance cutoff
        mask = distances > max_radius
        distances = np.ma.array(distances, mask=mask)

        # Sort neighbors by distance
        neighbor_ranks = stats.rankdata(
            distances,
            method='ordinal',
            axis=1
        )

        # Apply max neighbors cutoff
        neighbor_ranks[neighbor_ranks > max_neighbors] = 0
        neighbor_mask = neighbor_ranks > 0

        # Get edges and weights
        edges = np.nonzero(neighbor_mask)
        edge_weights = distances[edges]

        return edges, edge_weights

    def _compute_node_features(
            self,
            structure: ase.Atoms
    ) -> np.ndarray:
        """Compute node features from atomic numbers."""
        atomic_numbers = structure.get_atomic_numbers()

        # Get feature vectors for each atom
        features = np.array([
            self.atom_features[str(Z)] for Z in atomic_numbers
        ])

        # Add degree features if requested
        if self.processing_args.get("use_degree_features", True):
            degree_features = self._compute_degree_features(
                len(structure),
                self.processing_args["graph_max_neighbors"]
            )
            features = np.concatenate([features, degree_features], axis=1)

        return features.astype(np.float32)

    def _compute_degree_features(
            self,
            num_nodes: int,
            max_degree: int
    ) -> np.ndarray:
        """Compute one-hot degree features."""
        degrees = np.zeros((num_nodes, max_degree + 1))
        for i, degree in enumerate(self.graphs[-1].get_degrees()):
            if degree <= max_degree:
                degrees[i, degree] = 1
        return degrees

    def _compute_edge_features(
            self,
            structure: ase.Atoms,
            graph: GraphData
    ) -> np.ndarray:
        """Compute edge features."""
        # Normalize distances
        distances = self._normalize_distances(graph.edge_weights)

        # Generate Gaussian basis
        gbs = GaussianBasis(
            start=0.0,
            stop=1.0,
            num_gaussians=self.processing_args["graph_edge_length"],
            width=0.2
        )
        edge_features = gbs(distances)

        return edge_features

    def _compute_global_features(
            self,
            structure: ase.Atoms
    ) -> np.ndarray:
        """Compute global structure features."""
        # Composition features
        atomic_numbers = structure.get_atomic_numbers()
        composition = np.zeros(108)  # Max atomic number
        unique, counts = np.unique(atomic_numbers, return_counts=True)
        composition[unique] = counts / len(atomic_numbers)

        # Can add additional global features here

        return composition.astype(np.float32)

    def _normalize_distances(
            self,
            distances: np.ndarray
    ) -> np.ndarray:
        """Normalize edge distances to [0,1] range."""
        if not hasattr(self, 'distance_scaler'):
            self.distance_scaler = StandardScaler()
            self.distance_scaler.fit(distances.reshape(-1, 1))

        normalized = self.distance_scaler.transform(
            distances.reshape(-1, 1)
        ).reshape(-1)

        # Clip to [0,1] range
        normalized = np.clip(
            (normalized - normalized.min()) /
            (normalized.max() - normalized.min()),
            0, 1
        )

        return normalized

    def _save_processed_data(self):
        """Save processed graph data."""
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Save as single file or multiple based on size
        if len(self.graphs) * self.graphs[0].memory_size < 1e9:  # 1 GB threshold
            self.logger.info("Saving as single file")
            tf.saved_model.save(
                self.graphs,
                str(self.processed_path / "data")
            )
        else:
            self.logger.info("Saving as multiple files")
            for i, graph in enumerate(self.graphs):
                tf.saved_model.save(
                    graph,
                    str(self.processed_path / f"data_{i}")
                )

    def load(self):
        """Load processed data."""
        if not self.processed_path.exists():
            self.logger.info("Processing data first")
            self.process()
            return

        # Load processed data
        if (self.processed_path / "data").exists():
            self.graphs = tf.saved_model.load(
                str(self.processed_path / "data")
            )
        else:
            self.graphs = []
            for path in sorted(self.processed_path.glob("data_*")):
                self.graphs.append(
                    tf.saved_model.load(str(path))
                )

        self.processed = True

    def get_dataloader(
            self,
            batch_size: int,
            shuffle: bool = True,
            drop_last: bool = False
    ) -> tf.data.Dataset:
        """Create TensorFlow data loader."""
        if not self.processed:
            self.load()

        # Convert to TF dataset
        dataset = tf.data.Dataset.from_generator(
            lambda: self.graphs,
            output_signature=GraphData.spec()
        )

        # Apply batching
        if shuffle:
            dataset = dataset.shuffle(len(self.graphs))

        dataset = dataset.batch(
            batch_size,
            drop_remainder=drop_last
        )

        return dataset

    def split_data(
            self,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            seed: Optional[int] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Split data into train/val/test sets."""
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

        # Set random seed
        if seed is not None:
            tf.random.set_seed(seed)

        # Create full dataset
        dataset = self.get_dataloader(
            batch_size=1,
            shuffle=True
        )

        # Calculate split sizes
        total_size = len(self.graphs)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)

        # Split dataset
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)

        return train_dataset, val_dataset, test_dataset


class GaussianBasis(tf.keras.layers.Layer):
    """
    Gaussian basis expansion for distance features.

    Args:
        start: Minimum distance for centers
        stop: Maximum distance for centers
        num_gaussians: Number of Gaussian functions
        width: Width of Gaussian functions
    """

    def __init__(
            self,
            start: float = 0.0,
            stop: float = 5.0,
            num_gaussians: int = 50,
            width: float = 0.5,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        self.width = width

        # Initialize centers
        centers = np.linspace(start, stop, num_gaussians)
        self.centers = tf.constant(centers, dtype=tf.float32)

        # Initialize width
        self.coeff = -0.5 / ((stop - start) * width) ** 2

    def call(self, distances: tf.Tensor) -> tf.Tensor:
        """Expand distances in Gaussian basis."""
        # Reshape for broadcasting
        distances = tf.expand_dims(distances, -1)
        centers = tf.reshape(self.centers, [1, -1])

        # Compute Gaussian features
        diff = distances - centers
        return tf.exp(self.coeff * tf.square(diff))

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'start': self.start,
            'stop': self.stop,
            'num_gaussians': self.num_gaussians,
            'width': self.width
        })
        return config

class ChemicalFeatureComputer:
    """
    Advanced chemical feature computation for materials.

    Features:
    - SOAP descriptors
    - Coulomb matrices
    - Sine matrices
    - Voronoi analysis

    Args:
        species: List of chemical species
        processing_args: Feature computation parameters
    """

    def __init__(
            self,
            species: List[str],
            processing_args: Dict[str, Any]
    ):
        self.species = species
        self.processing_args = processing_args
        self.logger = logging.getLogger(__name__)

    def compute_soap(
            self,
            structure: ase.Atoms
    ) -> np.ndarray:
        """
        Compute SOAP (Smooth Overlap of Atomic Positions) features.

        Args:
            structure: ASE atoms object

        Returns:
            SOAP descriptor array
        """
        from dscribe.descriptors import SOAP

        # Set up SOAP calculator
        soap = SOAP(
            species=self.species,
            rcut=self.processing_args["SOAP_rcut"],
            nmax=self.processing_args["SOAP_nmax"],
            lmax=self.processing_args["SOAP_lmax"],
            sigma=self.processing_args["SOAP_sigma"],
            periodic=any(structure.pbc),
            sparse=False,
            average="inner",
            rbf="gto",
            crossover=False
        )

        # Compute features
        features = soap.create(structure)
        return features.astype(np.float32)

    def compute_sine_matrix(
            self,
            structure: ase.Atoms,
            n_atoms_max: int
    ) -> np.ndarray:
        """
        Compute Sine matrix descriptor for periodic systems.

        Args:
            structure: ASE atoms object
            n_atoms_max: Maximum number of atoms

        Returns:
            Sine matrix descriptor
        """
        from dscribe.descriptors import SineMatrix

        # Set up calculator
        sm = SineMatrix(
            n_atoms_max=n_atoms_max,
            permutation="eigenspectrum",
            sparse=False,
            flatten=True
        )

        # Compute features
        features = sm.create(structure)
        return features.astype(np.float32)

    def compute_coulomb_matrix(
            self,
            structure: ase.Atoms,
            n_atoms_max: int
    ) -> np.ndarray:
        """
        Compute Coulomb matrix descriptor for non-periodic systems.

        Args:
            structure: ASE atoms object
            n_atoms_max: Maximum number of atoms

        Returns:
            Coulomb matrix descriptor
        """
        from dscribe.descriptors import CoulombMatrix

        # Set up calculator
        cm = CoulombMatrix(
            n_atoms_max=n_atoms_max,
            permutation="eigenspectrum",
            sparse=False,
            flatten=True
        )

        # Compute features
        features = cm.create(structure)
        return features.astype(np.float32)

    def compute_voronoi(
            self,
            structure: ase.Atoms,
            max_neighbors: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Voronoi tessellation and connectivity.

        Args:
            structure: ASE atoms object
            max_neighbors: Maximum number of neighbors

        Returns:
            Tuple of (edges, weights) from Voronoi analysis
        """
        from pymatgen.core.structure import Structure
        from pymatgen.analysis.structure_analyzer import VoronoiConnectivity

        # Convert to pymatgen structure
        structure_pmg = Structure.from_sites(structure)

        # Compute Voronoi connectivity
        voronoi = VoronoiConnectivity(
            structure_pmg,
            cutoff=self.processing_args["graph_max_radius"]
        )
        connections = voronoi.connectivity_array

        # Get edges and weights
        edges, weights = self._threshold_sort(
            connections,
            max_neighbors,
            reverse=True
        )

        return edges.astype(np.int64), weights.astype(np.float32)

    def _threshold_sort(
            self,
            matrix: np.ndarray,
            max_neighbors: int,
            reverse: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sort and threshold connectivity matrix."""
        # Rank entries
        if reverse:
            ranked = stats.rankdata(
                -matrix,
                method='ordinal',
                axis=1
            )
        else:
            ranked = stats.rankdata(
                matrix,
                method='ordinal',
                axis=1
            )

        # Apply neighbor cutoff
        mask = ranked <= max_neighbors

        # Get edges and weights
        edges = np.nonzero(mask)
        weights = matrix[edges]

        return edges, weights


class DataAugmentation:
    """
    Data augmentation for materials graphs.

    Features:
    - Random rotations
    - Random perturbations
    - Edge masking
    - Node feature noise

    Args:
        processing_args: Augmentation parameters
    """

    def __init__(
            self,
            processing_args: Dict[str, Any]
    ):
        self.processing_args = processing_args

    def rotate_structure(
            self,
            structure: ase.Atoms
    ) -> ase.Atoms:
        """Apply random rotation to structure."""
        # Get random rotation matrix
        angles = np.random.uniform(0, 2 * np.pi, 3)
        Rx = self._rotation_matrix(angles[0], [1, 0, 0])
        Ry = self._rotation_matrix(angles[1], [0, 1, 0])
        Rz = self._rotation_matrix(angles[2], [0, 0, 1])
        R = Rz @ Ry @ Rx

        # Apply rotation
        positions = structure.get_positions()
        rotated_positions = positions @ R

        # Create new structure
        new_structure = structure.copy()
        new_structure.set_positions(rotated_positions)

        return new_structure

    def perturb_positions(
            self,
            structure: ase.Atoms,
            sigma: float = 0.01
    ) -> ase.Atoms:
        """Add random perturbations to atomic positions."""
        positions = structure.get_positions()
        noise = np.random.normal(0, sigma, positions.shape)

        new_structure = structure.copy()
        new_structure.set_positions(positions + noise)

        return new_structure

    def mask_edges(
            self,
            graph: GraphData,
            mask_ratio: float = 0.1
    ) -> GraphData:
        """Randomly mask edges."""
        num_edges = len(graph.edges[0])
        mask_size = int(mask_ratio * num_edges)

        # Select random edges to mask
        mask_indices = np.random.choice(
            num_edges,
            mask_size,
            replace=False
        )

        # Create new graph with masked edges
        new_graph = graph.copy()
        mask = np.ones(num_edges, dtype=bool)
        mask[mask_indices] = False

        new_graph.edges = graph.edges[:, mask]
        new_graph.edge_weights = graph.edge_weights[mask]
        if hasattr(graph, 'edge_features'):
            new_graph.edge_features = graph.edge_features[mask]

        return new_graph

    def add_feature_noise(
            self,
            graph: GraphData,
            sigma: float = 0.01
    ) -> GraphData:
        """Add Gaussian noise to node features."""
        new_graph = graph.copy()
        noise = np.random.normal(0, sigma, graph.node_features.shape)
        new_graph.node_features = graph.node_features + noise
        return new_graph

    def _rotation_matrix(
            self,
            angle: float,
            axis: List[float]
    ) -> np.ndarray:
        """Generate 3D rotation matrix."""
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)

        return np.array([
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]
        ])


def setup_processing(
        job_parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Set up data processing configuration."""
    # Default processing parameters
    defaults = {
        "data_format": "cif",
        "dictionary_source": "default",
        "graph_max_radius": 6.0,
        "graph_max_neighbors": 12,
        "graph_edge_length": 50,
        "edge_features": True,
        "use_degree_features": True,
        "SOAP_descriptor": False,
        "SM_descriptor": False,
        "dataset_type": "inmemory",
        "verbose": True
    }

    # Update with job parameters
    processing_args = {**defaults, **job_parameters}

    # Validate parameters
    _validate_processing_args(processing_args)

    return processing_args


def _validate_processing_args(args: Dict[str, Any]):
    """Validate processing arguments."""
    required = [
        "data_format",
        "dictionary_source",
        "graph_max_radius",
        "graph_max_neighbors"
    ]

    for param in required:
        if param not in args:
            raise ValueError(f"Required parameter {param} not found")

    if args["graph_max_radius"] <= 0:
        raise ValueError("graph_max_radius must be positive")

    if args["graph_max_neighbors"] <= 0:
        raise ValueError("graph_max_neighbors must be positive")