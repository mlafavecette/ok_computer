"""
Super Message Passing Neural Network Implementation

An enhanced MPNN architecture with:
- Advanced message passing mechanisms
- GRU-based state updates
- Dynamic group normalization
- Residual connections
- Flexible pooling

Author: Claude
Date: 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class DynamicGroupNorm(layers.Layer):
    """Dynamic group normalization layer optimized for material properties."""

    def __init__(
            self,
            groups: int = 10,
            epsilon: float = 1e-5,
            momentum: float = 0.99,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = self.add_weight(
            'gamma',
            shape=(channels,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            'beta',
            shape=(channels,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        reshaped = tf.reshape(inputs, [-1, self.groups, input_shape[-1] // self.groups])

        mean, variance = tf.nn.moments(reshaped, axes=[2], keepdims=True)
        normalized = (reshaped - mean) / tf.sqrt(variance + self.epsilon)

        normalized = tf.reshape(normalized, input_shape)
        return normalized * self.gamma + self.beta


class EnhancedMessagePassing(layers.Layer):
    """Enhanced message passing with GRU updates and edge features."""

    def __init__(
            self,
            units: int,
            edge_dim: int,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units

        # Edge network
        self.edge_network = keras.Sequential([
            layers.Dense(units),
            layers.Activation(activation),
            layers.Dense(units * units)
        ])

        # GRU for state updates
        self.gru = layers.GRU(units, return_state=True)

        self.dropout = layers.Dropout(dropout_rate)

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
            training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Process inputs through message passing."""
        node_features, edge_features, edge_indices, hidden_state = inputs

        # Transform edge features
        edge_weights = self.edge_network(edge_features)
        edge_weights = tf.reshape(edge_weights, [-1, self.units, self.units])

        # Gather source node features
        source_features = tf.gather(node_features, edge_indices[0])

        # Compute messages
        messages = tf.einsum('bij,bj->bi', edge_weights, source_features)

        # Aggregate messages
        aggregated = tf.math.unsorted_segment_mean(
            messages,
            edge_indices[1],
            num_segments=tf.shape(node_features)[0]
        )

        # Update state with GRU
        aggregated = tf.expand_dims(aggregated, axis=0)
        output, new_state = self.gru(aggregated, initial_state=hidden_state)
        output = tf.squeeze(output, axis=0)
        output = self.dropout(output, training=training)

        return output, new_state

class SuperMPNN(keras.Model):
    """
    Enhanced Message Passing Neural Network for materials property prediction.

    Features:
    - Advanced message passing with edge features
    - GRU-based state updates
    - Dynamic group normalization
    - Residual connections
    - Flexible pooling strategies

    Args:
        num_features: Number of input node features
        num_edge_features: Number of edge features
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_message_layers: Number of message passing layers
        num_dense_layers: Number of dense layers
        activation: Activation function
        dropout_rate: Dropout rate
        pool_method: Graph pooling method ('mean', 'sum', 'max')
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
            self,
            num_features: int,
            num_edge_features: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_message_layers: int = 3,
            num_dense_layers: int = 2,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            pool_method: str = "mean",
            use_batch_norm: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Initial embeddings
        self.node_embedding = keras.Sequential([
            layers.Dense(hidden_dim),
            DynamicGroupNorm() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation),
            layers.Dropout(dropout_rate)
        ])

        self.edge_embedding = keras.Sequential([
            layers.Dense(hidden_dim),
            DynamicGroupNorm() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation),
            layers.Dropout(dropout_rate)
        ])

        # Message passing layers
        self.message_layers = [
            EnhancedMessagePassing(
                hidden_dim,
                hidden_dim,
                activation=activation,
                dropout_rate=dropout_rate
            ) for _ in range(num_message_layers)
        ]

        # Dense layers for output
        self.dense_layers = []
        for _ in range(num_dense_layers):
            self.dense_layers.extend([
                layers.Dense(hidden_dim),
                DynamicGroupNorm() if use_batch_norm else layers.Lambda(lambda x: x),
                layers.Activation(activation),
                layers.Dropout(dropout_rate)
            ])

        self.output_layer = layers.Dense(output_dim)
        self.pool_method = pool_method

    def pool_nodes(
            self,
            node_features: tf.Tensor,
            graph_indices: tf.Tensor
    ) -> tf.Tensor:
        """Pool node features to graph level using specified method."""
        if self.pool_method == "mean":
            return tf.math.segment_mean(node_features, graph_indices)
        elif self.pool_method == "sum":
            return tf.math.segment_sum(node_features, graph_indices)
        elif self.pool_method == "max":
            return tf.math.segment_max(node_features, graph_indices)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool_method}")

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Forward pass computation.

        Args:
            inputs: Tuple of:
                - node_features: [num_nodes, num_features]
                - edge_features: [num_edges, num_edge_features]
                - edge_indices: [2, num_edges]
                - graph_indices: [num_nodes]
            training: Whether in training mode

        Returns:
            Graph-level predictions [num_graphs, output_dim]
        """
        node_features, edge_features, edge_indices, graph_indices = inputs

        # Initial embeddings
        node_hidden = self.node_embedding(node_features, training=training)
        edge_hidden = self.edge_embedding(edge_features, training=training)

        # Initialize hidden state
        hidden_state = tf.zeros([1, tf.shape(node_features)[0], self.message_layers[0].units])

        # Store for residual connection
        residual = node_hidden

        # Message passing
        for layer in self.message_layers:
            node_out, hidden_state = layer(
                (node_hidden, edge_hidden, edge_indices, hidden_state),
                training=training
            )
            # Residual connection
            node_hidden = node_out + residual
            residual = node_hidden

        # Pool to graph level
        graph_features = self.pool_nodes(node_hidden, graph_indices)

        # Dense layers
        hidden = graph_features
        for layer in self.dense_layers:
            hidden = layer(hidden, training=training)

        # Final prediction
        return self.output_layer(hidden)

    def train_step(self, data):
        """
        Custom training step with gradient clipping.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dict of metric results
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        # Compute and clip gradients
        grads = tape.gradient(loss, self.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]

        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Custom test step.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dict of metric results
        """
        x, y = data
        y_pred = self(x, training=False)

        # Update metrics
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        """Returns model configuration."""
        return {
            "hidden_dim": self.message_layers[0].units,
            "num_message_layers": len(self.message_layers),
            "pool_method": self.pool_method
        }

    @classmethod
    def from_config(cls, config):
        """Creates model from configuration."""
        return cls(**config)

    def compute_output_shape(self, input_shape):
        """Computes output shape from input shape."""
        node_shape, _, _, graph_indices_shape = input_shape
        num_graphs = tf.math.maximum(graph_indices_shape[0], 0) + 1
        return (num_graphs, self.output_layer.units)
