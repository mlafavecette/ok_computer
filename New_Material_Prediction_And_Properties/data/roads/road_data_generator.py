import numpy as np
import pandas as pd
from ase import Atoms, Atom
from ase.io import write
import json
import h5py
from pathlib import Path
import random
from typing import Dict, List, Tuple
import itertools


class PlasticGridDataGenerator:
    def __init__(self, output_size_mb: int = 250):
        self.output_size_mb = output_size_mb
        self.materials = {
            'HDPE': {'proportion': 0.70, 'density': 0.97},
            'PP': {'proportion': 0.20, 'density': 0.91},
            'UV_inhibitor': {'proportion': 0.05, 'density': 1.05},
            'silica_coating': {'proportion': 0.05, 'density': 2.20}
        }
        self.grid_params = {
            'thickness': 2.5,  # inches
            'cell_size': 3.0,  # inches
            'load_capacity': 80000  # pounds per sq ft
        }

    def generate_composition_variation(self) -> Dict[str, float]:
        """Generate realistic variations in material compositions."""
        variation = {}
        for material, props in self.materials.items():
            # Add random variation within ±2% of target proportion
            base_prop = props['proportion']
            variation[material] = max(0, min(1, base_prop + np.random.normal(0, 0.02)))

        # Normalize to ensure proportions sum to 1
        total = sum(variation.values())
        return {k: v / total for k, v in variation.items()}

    def generate_crystal_structure(self) -> Atoms:
        """Generate simplified crystal structure for the composite material."""
        # Create a simplified unit cell
        cell_size = 10.0  # Angstroms
        positions = list(itertools.product(range(2), repeat=3))

        atoms = Atoms(
            symbols=['C'] * 4 + ['H'] * 4,  # Simplified polymer structure
            positions=[(p[0] * cell_size, p[1] * cell_size, p[2] * cell_size) for p in positions],
            cell=[cell_size] * 3,
            pbc=True
        )
        return atoms

    def generate_mechanical_properties(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Generate realistic mechanical properties based on composition."""
        base_properties = {
            'elastic_modulus': 800 + np.random.normal(0, 50),  # MPa
            'tensile_strength': 20 + np.random.normal(0, 2),  # MPa
            'impact_strength': 4 + np.random.normal(0, 0.5),  # kJ/m²
            'water_absorption': 0.01 + np.random.normal(0, 0.002),  # %
            'thermal_expansion': 1.2e-4 + np.random.normal(0, 1e-5)  # 1/K
        }

        # Adjust properties based on composition
        property_multipliers = {
            'HDPE': 1.0,
            'PP': 1.1,
            'UV_inhibitor': 0.95,
            'silica_coating': 1.15
        }

        return {
            prop: value * sum(comp * property_multipliers[mat]
                              for mat, comp in composition.items())
            for prop, value in base_properties.items()
        }

    def generate_sample(self) -> Dict:
        """Generate a single sample with all relevant properties."""
        composition = self.generate_composition_variation()
        crystal_structure = self.generate_crystal_structure()
        mechanical_props = self.generate_mechanical_properties(composition)

        return {
            'composition': composition,
            'crystal_structure': crystal_structure,
            'mechanical_properties': mechanical_props,
            'grid_params': self.grid_params.copy()
        }

    def save_dataset(self, num_samples: int, output_dir: str = 'plastic_grid_data'):
        """Save the generated dataset in multiple formats suitable for different ML models."""
        Path(output_dir).mkdir(exist_ok=True)

        # Save raw data in HDF5 format (suitable for VAE, SchNet)
        with h5py.File(f'{output_dir}/raw_data.h5', 'w') as f:
            for i in range(num_samples):
                sample = self.generate_sample()
                grp = f.create_group(f'sample_{i}')

                # Store composition
                for mat, prop in sample['composition'].items():
                    grp.create_dataset(f'composition/{mat}', data=prop)

                # Store crystal structure
                atoms = sample['crystal_structure']
                grp.create_dataset('positions', data=atoms.get_positions())
                grp.create_dataset('numbers', data=atoms.get_atomic_numbers())
                grp.create_dataset('cell', data=atoms.get_cell())

                # Store mechanical properties
                for prop, value in sample['mechanical_properties'].items():
                    grp.create_dataset(f'properties/{prop}', data=value)

        # Save graph representation (suitable for MPNN, GAT, GCN)
        graph_data = []
        for i in range(num_samples):
            sample = self.generate_sample()
            atoms = sample['crystal_structure']

            # Create graph representation
            graph = {
                'node_features': atoms.get_atomic_numbers(),
                'edge_index': self._get_edge_index(atoms),
                'edge_attr': self._get_edge_attributes(atoms),
                'global_features': list(sample['composition'].values()),
                'properties': list(sample['mechanical_properties'].values())
            }
            graph_data.append(graph)

        np.save(f'{output_dir}/graph_data.npy', graph_data)

        # Save descriptor-based representation (suitable for descriptor_nn)
        descriptors = []
        for i in range(num_samples):
            sample = self.generate_sample()
            descriptor = self._calculate_descriptors(sample)
            descriptors.append(descriptor)

        np.save(f'{output_dir}/descriptors.npy', descriptors)

    def _get_edge_index(self, atoms: Atoms) -> np.ndarray:
        """Generate edges for graph representation."""
        n_atoms = len(atoms)
        cutoff = 3.0  # Angstroms
        positions = atoms.get_positions()

        edges = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    edges.append([i, j])
                    edges.append([j, i])  # Add reverse edge for undirected graph

        return np.array(edges).T

    def _get_edge_attributes(self, atoms: Atoms) -> np.ndarray:
        """Calculate edge attributes (distances) for graph representation."""
        edges = self._get_edge_index(atoms)
        positions = atoms.get_positions()

        distances = []
        for i, j in edges.T:
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append([dist])

        return np.array(distances)

    def _calculate_descriptors(self, sample: Dict) -> np.ndarray:
        """Calculate chemical descriptors for the material."""
        composition = sample['composition']
        mechanical_props = sample['mechanical_properties']

        # Combine composition and mechanical properties into a fixed-length vector
        descriptor = np.concatenate([
            list(composition.values()),
            list(mechanical_props.values())
        ])

        return descriptor

    def generate(self):
        """Generate the complete dataset with the specified size."""
        # Calculate number of samples needed to reach target size
        sample_size_bytes = 1000  # Approximate size per sample
        num_samples = (self.output_size_mb * 1024 * 1024) // sample_size_bytes

        print(f"Generating {num_samples} samples...")
        self.save_dataset(num_samples)
        print("Dataset generation complete!")


if __name__ == "__main__":
    generator = PlasticGridDataGenerator(output_size_mb=250)
    generator.generate()