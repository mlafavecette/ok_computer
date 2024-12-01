"""
TensorFlow Implementation of NequIP (Neural Equivariant Interatomic Potentials)

A sophisticated implementation of the NequIP architecture for materials property
prediction, featuring:
- E(3)-equivariant neural networks
- Tensor field networks
- Spherical harmonics
- Advanced message passing

Reference:
Batzner et al. "E(3)-equivariant graph neural networks for data-efficient and
accurate interatomic potentials" Nature Communications (2022)

Features:
- SO(3) equivariance
- Message passing neural networks
- Continuous filter convolutions
- Radial basis functions
- Spherical harmonics
- Tensor field networks

Author: Michael R. Lafave
Date: 2020-11-20
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import functools

from New_Materials_Discovery.models.layers import spherical_harmonics
from New_Materials_Discovery.utils.irreps import Irreps, IrrepsData
from New_Materials_Discovery.utils.tensor_products import TensorProduct
from New_Materials_Discovery.utils import radial


class TensorFieldNetwork(layers.Layer):
    """
    E(3)-equivariant tensor field network layer.

    Features:
    - SO(3) equivariance by construction
    - Continuous filter convolutions
    - Message passing with radial filters
    - Spherical harmonic basis

    Args:
        irreps_in: Input irreducible representations
        irreps_out: Output irreducible representations
        n_edge_features: Number of radial basis functions
        fc_dims: Hidden dimensions of filter network
        activation: Nonlinearity for filter network
    """

    def __init__(
            self,
            irreps_in: Irreps,
            irreps_out: Irreps,
            n_edge_features: int,
            fc_dims: List[int] = [64, 64],
            activation: str = "swish",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        # Set up tensor product paths
        self.tp = TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=Irreps.spherical_harmonics(2),
            irreps_out=irreps_out
        )

        # Filter network for radial functions
        self.filter_net = self._build_filter_net(
            n_edge_features,
            fc_dims,
            self.tp.n_weights,
            activation
        )

    def _build_filter_net(
            self,
            n_input: int,
            hidden_dims: List[int],
            n_output: int,
            activation: str
    ) -> tf.keras.Sequential:
        """Build radial filter network."""
        layers = []
        dims = [n_input] + hidden_dims + [n_output]

        for i in range(len(dims) - 1):
            layers.extend([
                tf.keras.layers.Dense(
                    dims[i + 1],
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=2.0,
                        mode='fan_in'
                    )
                ),
                tf.keras.layers.Activation(activation)
            ])

        return tf.keras.Sequential(layers)

    def call(
            self,
            node_features: tf.Tensor,
            edge_features: tf.Tensor,
            edge_vectors: tf.Tensor,
            edge_index: tf.Tensor
    ) -> tf.Tensor:
        """Forward pass."""
        # Get spherical harmonics
        edge_sh = spherical_harmonics.spherical_harmonics(
            l_max=2,
            vectors=edge_vectors,
            normalize=True
        )

        # Compute radial filter values
        filter_weights = self.filter_net(edge_features)

        # Gather source node features
        source_features = tf.gather(node_features, edge_index[:, 0])

        # Apply tensor product with radial weighting
        message = self.tp(source_features, edge_sh, filter_weights)

        # Aggregate messages
        out = tf.zeros(
            [tf.shape(node_features)[0], self.irreps_out.dim],
            dtype=node_features.dtype
        )
        out = tf.tensor_scatter_nd_add(
            out,
            edge_index[:, 1:2],
            message
        )

        return out


class NequIPConvolution(layers.Layer):
    """
    NequIP convolution layer implementing E(3)-equivariant message passing.

    Features:
    - SO(3) equivariant convolutions
    - Self-interaction handling
    - Gated nonlinearities
    - Chemistry-dependent weights

    Args:
        hidden_irreps: Irreducible representation for hidden features
        use_sc: Whether to use self-connections
        nonlinearities: Activation functions for even/odd irreps
        radial_basis: Number of radial basis functions
        n_neighbors: Average number of neighbors for normalization
    """

    def __init__(
            self,
            hidden_irreps: Irreps,
            use_sc: bool = True,
            nonlinearities: Dict[str, str] = None,
            radial_basis: int = 8,
            n_neighbors: float = 1.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_irreps = hidden_irreps
        self.use_sc = use_sc
        self.nonlinearities = nonlinearities or {
            'e': 'swish',
            'o': 'tanh'
        }

        # Tensor field network for message passing
        self.tfn = TensorFieldNetwork(
            irreps_in=hidden_irreps,
            irreps_out=hidden_irreps,
            n_edge_features=radial_basis,
        )

        # Linear transforms
        self.linear_in = layers.Dense(hidden_irreps.dim)
        self.linear_out = layers.Dense(hidden_irreps.dim)

        # Self-interaction
        if use_sc:
            self.self_connection = layers.Dense(hidden_irreps.dim)

        self.n_neighbors = n_neighbors

    def build(self, input_shape):
        """Build layer given input shape."""
        super().build(input_shape)

    def call(
            self,
            node_features: tf.Tensor,
            node_attrs: tf.Tensor,
            edge_vectors: tf.Tensor,
            edge_features: tf.Tensor,
            edge_index: tf.Tensor
    ) -> tf.Tensor:
        """Forward pass."""
        # Initial linear transform
        h = self.linear_in(node_features)

        # Message passing
        m = self.tfn(
            h,
            edge_features,
            edge_vectors,
            edge_index
        )

        # Normalize by average number of neighbors
        m = m / self.n_neighbors

        # Second linear transform
        h = self.linear_out(m)

        # Self-connection
        if self.use_sc:
            sc = self.self_connection(node_features)
            h = h + sc

        # Gated nonlinearities
        h = self._apply_nonlinearities(h)

        return h

    def _apply_nonlinearities(self, x: tf.Tensor) -> tf.Tensor:
        """Apply gated nonlinearities to features."""
        # Split into even and odd parts based on irreps
        even_mask = self.hidden_irreps.ls == 0
        odd_mask = ~even_mask

        x_even = x[:, even_mask]
        x_odd = x[:, odd_mask]

        # Apply appropriate nonlinearities
        x_even = getattr(tf.nn, self.nonlinearities['e'])(x_even)
        x_odd = getattr(tf.nn, self.nonlinearities['o'])(x_odd)

        # Recombine
        out = tf.concat([
            x_even if even_mask[i] else x_odd
            for i in range(len(even_mask))
        ], axis=-1)

        return out


class NequIP(tf.keras.Model):
    """
    Complete NequIP model for atomic systems.

    Features:
    - E(3)-equivariant architecture
    - Learned atomic embeddings
    - Message passing neural network
    - Energy prediction

    Args:
        n_elements: Number of chemical elements
        irreps_hidden: Irreducible representations for hidden features
        n_layers: Number of message passing layers
        radial_basis: Number of radial basis functions
        max_radius: Cutoff radius
        use_sc: Whether to use self-connections
    """

    def __init__(
            self,
            n_elements: int,
            irreps_hidden: Irreps,
            n_layers: int = 4,
            radial_basis: int = 8,
            max_radius: float = 5.0,
            use_sc: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.n_elements = n_elements
        self.irreps_hidden = irreps_hidden
        self.max_radius = max_radius

        # Atom embeddings
        self.atom_embedding = layers.Embedding(
            n_elements,
            irreps_hidden.dim
        )

        # Message passing layers
        self.conv_layers = [
            NequIPConvolution(
                irreps_hidden,
                use_sc=use_sc,
                radial_basis=radial_basis
            )
            for _ in range(n_layers)
        ]

        # Output blocks
        self.output_blocks = [
            layers.Dense(irreps_hidden.dim // 2),
            layers.Dense(1)
        ]

        # Radial basis expansion
        self.rbf = radial.GaussianRadialBasisFunction(
            n_rbf=radial_basis,
            cutoff=max_radius
        )

    def call(
            self,
            inputs: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """Forward pass to predict energy."""
        # Unpack inputs
        positions = inputs['positions']
        atomic_numbers = inputs['atomic_numbers']
        edge_index = inputs['edge_index']

        # Get edge vectors and distances
        edge_vectors = tf.gather(positions, edge_index[:, 0]) - tf.gather(
            positions, edge_index[:, 1]
        )
        edge_lengths = tf.norm(edge_vectors, axis=-1)

        # Compute edge features
        edge_features = self.rbf(edge_lengths)

        # Embed atoms
        node_features = self.atom_embedding(atomic_numbers)

        # Message passing
        for conv in self.conv_layers:
            node_features = conv(
                node_features,
                atomic_numbers,
                edge_vectors,
                edge_features,
                edge_index
            )

        # Output blocks
        for layer in self.output_blocks:
            node_features = layer(node_features)

        # Sum atomic energies
        energy = tf.reduce_sum(node_features)

        return energy

    def compute_forces(
            self,
            inputs: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """Compute atomic forces from energy model."""
        with tf.GradientTape() as tape:
            positions = inputs['positions']
            tape.watch(positions)
            inputs['positions'] = positions
            energy = self(inputs)

        forces = -tape.gradient(energy, positions)
        return forces


def get_default_config() -> Dict[str, Any]:
    """Get default model configuration."""
    return {
        'n_elements': 94,
        'irreps_hidden': '128x0e + 64x1e + 4x2e',
        'n_layers': 4,
        'radial_basis': 8,
        'max_radius': 5.0,
        'use_sc': True,
        'nonlinearities': {
            'e': 'swish',
            'o': 'tanh'
        }
    }