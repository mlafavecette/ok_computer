"""
Crystal Graph Neural Network Implementation in TensorFlow

A sophisticated implementation of crystal structure graph neural networks
for materials property prediction.

Features:
- Graph neural networks for crystals
- Gaussian edge features
- Atomic embeddings
- Global state tracking
- Residual connections

Author: Michael R. Lafave
Date: 2022
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from ..utils.tensor_ops import segment_normalized
from ..utils.graph import GraphsTuple
from ..utils import irreps

# Constants from materials science domain knowledge
AVERAGE_NODE_COORDINATION = 9  # Average atomic coordination number
AVERAGE_EDGE_COORDINATION = 17  # Average edge coordination number


class BetaSwish(keras.layers.Layer):
    """Learnable Swish activation with beta parameter per feature."""

    def build(self, input_shape):
        features = input_shape[-1]
        self.beta = self.add_weight(
            'beta',
            shape=(features,),
            initializer='ones'
        )

    def call(self, inputs):
        return inputs * tf.sigmoid(self.beta * inputs)


class CrystalConvolution(keras.layers.Layer):
    """
    Crystal Graph Convolution Layer.

    Features:
    - Edge feature transformation
    - Node state updates
    - Global state tracking
    - Residual connections
    - Normalized aggregation

    Args:
        units: Number of output features
        activation: Nonlinearity to use
        residual: Whether to use residual connections
        aggregation: Method to aggregate messages
        normalization: Edge normalization method
    """

    def __init__(
            self,
            units: int,
            activation: str = 'swish',
            residual: bool = True,
            aggregation: str = 'mean',
            normalization: str = 'none',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.residual = residual

        # Transformations
        self.edge_transform = keras.Sequential([
            keras.layers.Dense(units),
            get_activation(activation),
            keras.layers.Dense(units)
        ])

        self.node_transform = keras.Sequential([
            keras.layers.Dense(units),
            get_activation(activation),
            keras.layers.Dense(units)
        ])

        self.global_transform = keras.Sequential([
            keras.layers.Dense(units),
            get_activation(activation),
            keras.layers.Dense(units)
        ])

        # Aggregation functions
        self.aggregation = get_aggregation(aggregation)
        self.normalization = get_normalization(normalization)

    def call(
            self,
            inputs: Tuple[tf.Tensor, ...],
            training: bool = None
    ) -> tf.Tensor:
        """Forward pass."""
        node_features, edge_features, senders, receivers, globals = inputs

        # Transform edge features
        edge_messages = self.edge_transform(
            tf.concat([
                edge_features,
                tf.gather(node_features, senders),
                tf.gather(node_features, receivers)
            ], axis=-1)
        )

        # Aggregate messages
        aggregated = self.aggregation(
            edge_messages,
            receivers,
            num_segments=tf.shape(node_features)[0]
        )

        # Apply normalization
        if self.normalization:
            aggregated = self.normalization(aggregated)

        # Transform node features
        node_hidden = self.node_transform(
            tf.concat([node_features, aggregated], axis=-1)
        )

        # Residual connection
        if self.residual:
            node_features = node_features + node_hidden
        else:
            node_features = node_hidden

        # Update global state
        global_hidden = self.global_transform(
            tf.concat([
                globals,
                tf.reduce_mean(node_features, axis=0, keepdims=True),
                tf.reduce_mean(edge_messages, axis=0, keepdims=True)
            ], axis=-1)
        )

        if self.residual:
            globals = globals + global_hidden
        else:
            globals = global_hidden

        return node_features, edge_features, globals


class CrystalGraphNN(keras.Model):
    """
    Complete Crystal Graph Neural Network.

    Features:
    - Multi-layer message passing
    - Global state tracking
    - Atomic embeddings
    - Energy prediction

    Args:
        num_elements: Number of chemical elements
        units: Size of hidden layers
        num_layers: Number of message passing layers
        activation: Nonlinearity to use
        residual: Whether to use residual connections
        aggregation: Message aggregation method
        normalization: Edge normalization method
    """

    def __init__(
            self,
            num_elements: int,
            units: int = 64,
            num_layers: int = 3,
            activation: str = 'swish',
            residual: bool = True,
            aggregation: str = 'mean',
            normalization: str = 'none',
            **kwargs
    ):
        super().__init__(**kwargs)

        # Embeddings
        self.atom_embedding = keras.layers.Embedding(
            num_elements,
            units
        )

        # Convolution layers
        self.conv_layers = [
            CrystalConvolution(
                units,
                activation=activation,
                residual=residual,
                aggregation=aggregation,
                normalization=normalization
            )
            for _ in range(num_layers)
        ]

        # Output blocks
        self.output_blocks = [
            keras.layers.Dense(units // 2, activation=activation),
            keras.layers.Dense(1)
        ]

    def call(
            self,
            inputs: GraphsTuple,
            training: bool = None
    ) -> tf.Tensor:
        """Forward pass to predict energy."""
        # Unpack inputs
        nodes = self.atom_embedding(inputs.nodes)
        edge_features = inputs.edges
        senders = inputs.senders
        receivers = inputs.receivers
        globals = inputs.globals

        # Message passing
        for conv in self.conv_layers:
            nodes, edge_features, globals = conv(
                (nodes, edge_features, senders, receivers, globals),
                training=training
            )

        # Predict per-atom energy contributions
        for layer in self.output_blocks:
            nodes = layer(nodes)

        # Sum to get total energy
        energy = tf.reduce_sum(nodes)

        return energy


# Utility functions
def get_activation(name: str) -> keras.layers.Layer:
    """Get activation function by name."""
    if name == 'swish':
        return BetaSwish()
    return keras.activations.get(name)


def get_aggregation(name: str) -> callable:
    """Get aggregation function by name."""
    if name == 'mean':
        return tf.math.segment_mean
    elif name == 'sum':
        return tf.math.segment_sum
    elif name == 'coordination':
        return segment_normalized(AVERAGE_EDGE_COORDINATION)
    raise ValueError(f'Unknown aggregation: {name}')


def get_normalization(name: str) -> Optional[callable]:
    """Get normalization function by name."""
    if name == 'none':
        return None
    elif name == 'nodes':
        return lambda x: x / AVERAGE_NODE_COORDINATION
    elif name == 'edges':
        return lambda x: x / AVERAGE_EDGE_COORDINATION
    raise ValueError(f'Unknown normalization: {name}')


def get_default_config() -> Dict[str, Any]:
    """Get default model configuration."""
    return {
        'num_elements': 94,
        'units': 64,
        'num_layers': 3,
        'activation': 'swish',
        'residual': True,
        'aggregation': 'mean',
        'normalization': 'none',
        'learning_rate': 1e-3,
        'weight_decay': 1e-5
    }