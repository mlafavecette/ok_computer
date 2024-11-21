"""
Message Passing Neural Network (MPNN) Implementation in TensorFlow

A sophisticated implementation of MPNN specifically optimized for materials science
applications. Features advanced message passing with edge networks and GRU-based updates.

Reference:
- Gilmer et al. "Neural Message Passing for Quantum Chemistry" (2017)

Features:
- Edge-conditioned message passing
- GRU-based state updates
- Advanced readout mechanisms
- Flexible pooling strategies
- Batch processing support

Author: Claude
Date: 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod


class EdgeNetwork(layers.Layer):
    """
    Neural network for computing edge-dependent messages.

    Transforms edge features into a weight matrix for message computation.

    Args:
        units: Hidden dimension
        message_units: Message dimension
        activation: Activation function
        use_bias: Whether to use bias terms
        kernel_initializer: Weight initialization method
    """

    def __init__(
            self,
            units: int,
            message_units: int,
            activation: str = "relu",
            use_bias: bool = True,
            kernel_initializer: str = "glorot_uniform",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.message_units = message_units

        self.transform = keras.Sequential([
            layers.Dense(
                units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer
            ),
            layers.Dense(
                units * message_units,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer
            )
        ])

    def call(self, edge_features: tf.Tensor) -> tf.Tensor:
        """
        Transform edge features into weight matrices.

        Args:
            edge_features: Edge features [num_edges, feature_dim]

        Returns:
            Weight matrices [num_edges, units, message_units]
        """
        weights = self.transform(edge_features)
        return tf.reshape(weights, [-1, self.units, self.message_units])


class MessageBlock(layers.Layer):
    """
    Message computation and aggregation block.

    Computes messages between nodes using edge-specific transformations
    and aggregates messages for each node.

    Args:
        units: Hidden dimension
        message_units: Message dimension
        message_activation: Message network activation
        update_activation: Update network activation
        aggregation: Message aggregation method
        batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate
    """

    def __init__(
            self,
            units: int,
            message_units: int,
            message_activation: str = "relu",
            update_activation: str = "tanh",
            aggregation: str = "mean",
            batch_norm: bool = True,
            dropout_rate: float = 0.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Message computation
        self.edge_network = EdgeNetwork(units, message_units, message_activation)

        # Message update
        self.gru = layers.GRU(units)

        # Regularization
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        self.dropout = layers.Dropout(dropout_rate)
        self.update_activation = layers.Activation(update_activation)

        self.aggregation = aggregation

    def compute_messages(
            self,
            node_features: tf.Tensor,
            edge_features: tf.Tensor,
            edge_indices: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute messages between nodes.

        Args:
            node_features: Node features [num_nodes, feature_dim]
            edge_features: Edge features [num_edges, edge_dim]
            edge_indices: Edge connectivity [2, num_edges]

        Returns:
            Messages for each edge [num_edges, message_dim]
        """
        # Get source node features
        source_features = tf.gather(node_features, edge_indices[0])

        # Transform edge features to weight matrices
        edge_weights = self.edge_network(edge_features)

        # Compute messages
        messages = tf.einsum('bij,bj->bi', edge_weights, source_features)
        return messages

    def aggregate_messages(
            self,
            messages: tf.Tensor,
            node_indices: tf.Tensor,
            num_nodes: int
    ) -> tf.Tensor:
        """
        Aggregate messages for each node.

        Args:
            messages: Edge messages [num_edges, message_dim]
            node_indices: Node indices for each edge
            num_nodes: Total number of nodes

        Returns:
            Aggregated messages [num_nodes, message_dim]
        """
        if self.aggregation == "mean":
            return tf.math.unsorted_segment_mean(messages, node_indices, num_nodes)
        elif self.aggregation == "sum":
            return tf.math.unsorted_segment_sum(messages, node_indices, num_nodes)
        elif self.aggregation == "max":
            return tf.math.unsorted_segment_max(messages, node_indices, num_nodes)
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation}")

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Forward pass computation.

        Args:
            inputs: Tuple of (node_features, edge_features, edge_indices, node_state)
            training: Whether in training mode

        Returns:
            Updated node features
        """
        node_features, edge_features, edge_indices, node_state = inputs

        # Compute and aggregate messages
        messages = self.compute_messages(node_features, edge_features, edge_indices)
        aggregated = self.aggregate_messages(
            messages,
            edge_indices[1],
            tf.shape(node_features)[0]
        )

        # Update node states using GRU
        aggregated = tf.expand_dims(aggregated, axis=0)
        node_state = tf.expand_dims(node_state, axis=0)
        output, state = self.gru(aggregated, initial_state=node_state)
        output = tf.squeeze(output, axis=0)

        # Apply regularization
        if self.batch_norm is not None:
            output = self.batch_norm(output, training=training)
        output = self.dropout(output, training=training)
        output = self.update_activation(output)

        return output, tf.squeeze(state, axis=0)


class MPNN(keras.Model):
    """
    Complete Message Passing Neural Network for materials property prediction.

    Features multiple message passing layers with GRU-based updates and
    sophisticated readout mechanisms.

    Args:
        num_features: Number of input node features
        num_edge_features: Number of edge features
        hidden_dim: Hidden layer dimension
        message_dim: Message dimension
        output_dim: Output dimension
        num_message_layers: Number of message passing layers
        num_readout_layers: Number of readout layers
        batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate
        aggregation: Message aggregation method
        pool_method: Graph pooling method
    """

    def __init__(
            self,
            num_features: int,
            num_edge_features: int,
            hidden_dim: int = 64,
            message_dim: int = 64,
            output_dim: int = 1,
            num_message_layers: int = 3,
            num_readout_layers: int = 2,
            batch_norm: bool = True,
            dropout_rate: float = 0.0,
            aggregation: str = "mean",
            pool_method: str = "mean",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.pool_method = pool_method

        # Initial embeddings
        self.node_embedding = keras.Sequential([
            layers.Dense(hidden_dim),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation("relu"),
            layers.Dropout(dropout_rate)
        ])

        self.edge_embedding = keras.Sequential([
            layers.Dense(hidden_dim),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation("relu"),
            layers.Dropout(dropout_rate)
        ])

        # Message passing layers
        self.message_layers = [
            MessageBlock(
                hidden_dim,
                message_dim,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                aggregation=aggregation
            ) for _ in range(num_message_layers)
        ]

        # Readout network
        readout_layers = []
        for _ in range(num_readout_layers):
            readout_layers.extend([
                layers.Dense(hidden_dim),
                layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
                layers.Activation("relu"),
                layers.Dropout(dropout_rate)
            ])
        readout_layers.append(layers.Dense(output_dim))
        self.readout_network = keras.Sequential(readout_layers)

    def pool_nodes(self, node_features: tf.Tensor, graph_indices: tf.Tensor) -> tf.Tensor:
        """Pools node features to graph level using specified method."""
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
            inputs: Tuple of (node_features, edge_features, edge_indices, graph_indices)
            training: Whether in training mode

        Returns:
            Graph-level predictions
        """
        node_features, edge_features, edge_indices, graph_indices = inputs

        # Initial embeddings
        node_hidden = self.node_embedding(node_features, training=training)
        edge_hidden = self.edge_embedding(edge_features, training=training)

        # Initialize node states
        node_state = node_hidden

        # Message passing
        for layer in self.message_layers:
            node_hidden, node_state = layer(
                (node_hidden, edge_hidden, edge_indices, node_state),
                training=training
            )

        # Pool to graph level
        graph_features = self.pool_nodes(node_hidden, graph_indices)

        # Readout
        return self.readout_network(graph_features, training=training)

    def get_config(self):
        """Returns model configuration."""
        return {
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "pool_method": self.pool_method
        }