"""
E(3)-Equivariant Neural Network Layers in TensorFlow

A sophisticated implementation of E(3)-equivariant layers for materials science,
featuring:
- SO(3) equivariant operations
- Tensor product layers
- Spherical harmonics
- Irreducible representations

Features:
- Full E(3) equivariance
- Tensor field networks
- Advanced message passing
- Continuous convolutions

Author:Michael R. Lafave
Date: 2021
License: Apache 2.0
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import einops

from ..utils.irreps import Irreps, IrrepsArray
from ..utils.tensor_products import TensorProduct
from ..utils.spherical_harmonics import SphericalHarmonics


class FullyConnectedTensorProduct(keras.layers.Layer):
    """
    E(3)-equivariant fully-connected tensor product layer.

    Features:
    - SO(3) equivariance by construction
    - Learned weights for each path
    - Flexible irrep combinations
    - Efficient tensor operations

    Args:
        irreps_in1: Input irreps 1
        irreps_in2: Input irreps 2
        irreps_out: Output irreps
        normalize: Whether to normalize outputs
        internal_weights: Whether to use internal weights
    """

    def __init__(
            self,
            irreps_in1: Union[str, Irreps],
            irreps_in2: Union[str, Irreps],
            irreps_out: Union[str, Irreps],
            normalize: bool = True,
            internal_weights: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Convert irreps specifications
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)

        self.normalize = normalize
        self.internal_weights = internal_weights

        # Set up tensor product paths
        self.tp = TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out
        )

        if internal_weights:
            # Initialize learnable weights for each path
            self.weights = [
                self.add_weight(
                    name=f"weight_{i}",
                    shape=path.weight_shape,
                    initializer=keras.initializers.TruncatedNormal(
                        stddev=path.weight_std
                    )
                )
                for i, path in enumerate(self.tp.paths)
            ]

    def call(
            self,
            inputs1: tf.Tensor,
            inputs2: tf.Tensor,
            weights: Optional[List[tf.Tensor]] = None
    ) -> tf.Tensor:
        """Forward pass."""
        # Use provided or internal weights
        if not self.internal_weights:
            if weights is None:
                raise ValueError("External weights must be provided when internal_weights=False")
        else:
            weights = self.weights

        # Compute tensor product for each path
        outputs = []
        for path, weight in zip(self.tp.paths, weights):
            # Gather relevant input features
            x1 = tf.gather(inputs1, path.in1_indices, axis=-1)
            x2 = tf.gather(inputs2, path.in2_indices, axis=-1)

            # Reshape for broadcasting
            x1 = einops.rearrange(x1, 'b ... i -> b ... 1 i')
            x2 = einops.rearrange(x2, 'b ... j -> b ... j 1')

            # Compute tensor product
            prod = path.clebsch_gordan_tensor * x1 * x2

            # Apply weights
            if weight is not None:
                weight = einops.rearrange(weight, 'i j k -> 1 1 i j k')
                prod = tf.einsum('...ijk,ijkl->...l', prod, weight)

            # Aggregate over indices
            out = tf.reduce_sum(prod, axis=[-2, -3])

            outputs.append(out)

        # Concatenate outputs
        output = tf.concat(outputs, axis=-1)

        # Normalize if requested
        if self.normalize:
            output = output / np.sqrt(output.shape[-1])

        return IrrepsArray(self.irreps_out, output)

    def get_config(self):
        """Return layer configuration."""
        return {
            "irreps_in1": str(self.irreps_in1),
            "irreps_in2": str(self.irreps_in2),
            "irreps_out": str(self.irreps_out),
            "normalize": self.normalize,
            "internal_weights": self.internal_weights
        }


class Linear(keras.layers.Layer):
    """
    E(3)-equivariant linear layer.

    Features:
    - SO(3) equivariant linear transform
    - Optional bias term
    - Flexible irrep handling
    - Efficient implementation

    Args:
        irreps_in: Input irreps
        irreps_out: Output irreps
        bias: Whether to use bias
        normalize: Whether to normalize outputs
    """

    def __init__(
            self,
            irreps_in: Union[str, Irreps],
            irreps_out: Union[str, Irreps],
            bias: bool = True,
            normalize: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.use_bias = bias
        self.normalize = normalize

        # Initialize weights
        self.weights = []
        self.biases = []

        for mul_in, ir_in in self.irreps_in:
            for mul_out, ir_out in self.irreps_out:
                if ir_in.l == ir_out.l:  # Only connect same l
                    weight = self.add_weight(
                        f"weight_l{ir_in.l}",
                        shape=(mul_in, mul_out),
                        initializer=keras.initializers.TruncatedNormal(
                            stddev=1 / np.sqrt(mul_in)
                        )
                    )
                    self.weights.append((weight, ir_in.l))

        if bias:
            for mul, ir in self.irreps_out:
                if ir.l == 0:  # Only bias scalars
                    bias = self.add_weight(
                        f"bias_l{ir.l}",
                        shape=(mul,),
                        initializer="zeros"
                    )
                    self.biases.append(bias)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        outputs = []

        # Apply weights
        for weight, l in self.weights:
            # Get relevant input features
            x = inputs[..., self.irreps_in.get_indices(l)]

            # Linear transform
            out = tf.matmul(x, weight)
            outputs.append(out)

        # Concatenate outputs
        output = tf.concat(outputs, axis=-1)

        # Add biases
        if self.use_bias:
            for bias in self.biases:
                idx = self.irreps_out.get_indices(0)
                output[..., idx] += bias

        # Normalize if requested
        if self.normalize:
            output = output / np.sqrt(output.shape[-1])

        return IrrepsArray(self.irreps_out, output)

    def get_config(self):
        """Return layer configuration."""
        return {
            "irreps_in": str(self.irreps_in),
            "irreps_out": str(self.irreps_out),
            "bias": self.use_bias,
            "normalize": self.normalize
        }


class TensorFieldNetwork(keras.layers.Layer):
    """
    Tensor Field Network layer.

    Features:
    - E(3)-equivariant convolutions
    - Continuous filters
    - Multi-scale processing
    - Efficient implementation

    Args:
        irreps_in: Input irreps
        irreps_out: Output irreps
        num_bases: Number of radial bases
        activation: Nonlinearity to use
    """

    def __init__(
            self,
            irreps_in: Union[str, Irreps],
            irreps_out: Union[str, Irreps],
            num_bases: int = 10,
            activation: str = "silu",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)

        # Spherical harmonics calculator
        self.spherical_harmonics = SphericalHarmonics(
            max_l=max(self.irreps_out.ls)
        )

        # Radial network
        self.radial_network = keras.Sequential([
            keras.layers.Dense(32, activation=activation),
            keras.layers.Dense(32, activation=activation),
            keras.layers.Dense(num_bases)
        ])

        # Tensor product
        self.tensor_product = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=self.spherical_harmonics.irreps,
            irreps_out=irreps_out
        )

    def call(
            self,
            node_features: tf.Tensor,
            edge_vectors: tf.Tensor,
            edge_lengths: tf.Tensor,
            edge_index: tf.Tensor
    ) -> tf.Tensor:
        """Forward pass."""
        # Compute spherical harmonics
        edge_sh = self.spherical_harmonics(
            vectors=edge_vectors,
            normalize=True
        )

        # Compute radial weights
        radial_weights = self.radial_network(
            tf.expand_dims(edge_lengths, -1)
        )

        # Gather source features
        source_features = tf.gather(
            node_features,
            edge_index[:, 0]
        )

        # Apply tensor product
        messages = self.tensor_product(
            source_features,
            edge_sh,
            radial_weights
        )

        # Aggregate messages
        output = tf.zeros(
            [tf.shape(node_features)[0], self.irreps_out.dim],
            dtype=node_features.dtype
        )
        output = tf.tensor_scatter_nd_add(
            output,
            edge_index[:, 1:2],
            messages
        )

        return output

    def get_config(self):
        """Return layer configuration."""
        return {
            "irreps_in": str(self.irreps_in),
            "irreps_out": str(self.irreps_out),
            "num_bases": self.num_bases,
            "activation": self.activation
        }