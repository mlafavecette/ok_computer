"""
Materials Enriched Graph Neural Network (MEGNet) Implementation in TensorFlow

A sophisticated implementation of the MEGNet architecture specifically designed for
materials science applications. Features hierarchical message passing between
atomic, edge, and global state representations.

Reference:
- Chen et al. "Graph Networks as a Universal Machine Learning Framework for
  Molecules and Crystals" (2019)

Features:
- Multi-level message passing (atomic, bond, state)
- Edge feature integration
- Global state tracking
- Advanced normalization
- Residual connections

Author: Claude
Date: 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from functools import partial


class MEGNetBlock(layers.Layer):
    """
    Core MEGNet block implementing message passing between nodes, edges and global state.

    Implements three levels of information propagation:
    1. Edge updates considering adjacent nodes and global state
    2. Node updates aggregating edge messages and global state
    3. Global state updates from aggregated node and edge information
    """

    def __init__(
            self,
            units: int,
            activation: str = "relu",
            batch_norm: bool = True,
            dropout_rate: float = 0.0,
            kernel_initializer: str = "glorot_uniform",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units

        # Edge update network
        self.edge_network = self._build_update_network(
            units,
            4 * units,  # [source, target, edge, global]
            activation,
            batch_norm,
            dropout_rate,
            kernel_initializer
        )

        # Node update network
        self.node_network = self._build_update_network(
            units,
            3 * units,  # [node, edge_msg, global]
            activation,
            batch_norm,
            dropout_rate,
            kernel_initializer
        )

        # Global update network
        self.global_network = self._build_update_network(
            units,
            3 * units,  # [node_msg, edge_msg, global]
            activation,
            batch_norm,
            dropout_rate,
            kernel_initializer
        )

    def _build_update_network(
            self,
            units: int,
            input_dim: int,
            activation: str,
            batch_norm: bool,
            dropout_rate: float,
            kernel_initializer: str
    ) -> keras.Sequential:
        """Builds an update network with specified configuration."""
        layers_list = []

        # Input projection
        layers_list.extend([
            layers.Dense(
                units,
                kernel_initializer=kernel_initializer,
                use_bias=not batch_norm
            ),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation),
            layers.Dropout(dropout_rate)
        ])

        # Additional processing layers
        layers_list.extend([
            layers.Dense(
                units,
                kernel_initializer=kernel_initializer,
                use_bias=not batch_norm
            ),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation),
            layers.Dropout(dropout_rate)
        ])

        return keras.Sequential(layers_list)

    def _aggregate_edge_messages(
            self,
            edge_features: tf.Tensor,
            node_indices: tf.Tensor,
            num_nodes: int
    ) -> tf.Tensor:
        """Aggregates edge messages for each node."""
        return tf.math.unsorted_segment_mean(
            edge_features,
            node_indices,
            num_segments=num_nodes
        )

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
            training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Forward pass computation.

        Args:
            inputs: Tuple of (node_features, edge_features, edge_indices,
                   graph_indices, global_features)
            training: Whether in training mode

        Returns:
            Updated (node_features, edge_features, global_features)
        """
        node_features, edge_features, edge_indices, graph_indices, global_features = inputs

        # Get source and target node features for each edge
        source_features = tf.gather(node_features, edge_indices[:, 0])
        target_features = tf.gather(node_features, edge_indices[:, 1])

        # Get global features for each edge's graph
        edge_graph_features = tf.gather(global_features, tf.gather(graph_indices, edge_indices[:, 0]))

        # Update edges
        edge_inputs = tf.concat([
            source_features,
            target_features,
            edge_features,
            edge_graph_features
        ], axis=-1)
        edge_features_updated = self.edge_network(edge_inputs, training=training)

        # Aggregate edge messages for nodes
        edge_messages = self._aggregate_edge_messages(
            edge_features_updated,
            edge_indices[:, 0],
            tf.shape(node_features)[0]
        )

        # Update nodes
        node_graph_features = tf.gather(global_features, graph_indices)
        node_inputs = tf.concat([
            node_features,
            edge_messages,
            node_graph_features
        ], axis=-1)
        node_features_updated = self.node_network(node_inputs, training=training)

        # Aggregate node and edge messages for global
        node_messages = tf.math.segment_mean(node_features_updated, graph_indices)
        edge_messages_global = tf.math.segment_mean(
            edge_features_updated,
            tf.gather(graph_indices, edge_indices[:, 0])
        )

        # Update global
        global_inputs = tf.concat([
            node_messages,
            edge_messages_global,
            global_features
        ], axis=-1)
        global_features_updated = self.global_network(global_inputs, training=training)

        return node_features_updated, edge_features_updated, global_features_updated


class MEGNet(keras.Model):
    """
    Complete MEGNet architecture for materials property prediction.

    Combines multiple MEGNet blocks with sophisticated embedding and readout phases.
    Designed specifically for learning from atomic structures with edge features
    and global state information.

    Args:
        num_features: Number of input node features
        num_edge_features: Number of edge features
        num_global_features: Number of global features
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_blocks: Number of MEGNet blocks
        num_fc_layers: Number of fully connected layers
        batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate
        activation: Activation function
        pool_method: Graph pooling method
        learning_rate: Initial learning rate
    """

    def __init__(
            self,
            num_features: int,
            num_edge_features: int,
            num_global_features: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_blocks: int = 3,
            num_fc_layers: int = 2,
            batch_norm: bool = True,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            pool_method: str = "mean",
            learning_rate: float = 0.001,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Initial embeddings
        self.node_embedding = keras.Sequential([
            layers.Dense(hidden_dim),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation),
            layers.Dense(hidden_dim),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation)
        ])

        self.edge_embedding = keras.Sequential([
            layers.Dense(hidden_dim),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation),
            layers.Dense(hidden_dim),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation)
        ])

        self.global_embedding = keras.Sequential([
            layers.Dense(hidden_dim),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation),
            layers.Dense(hidden_dim),
            layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
            layers.Activation(activation)
        ])

        # MEGNet blocks
        self.meg_blocks = [
            MEGNetBlock(
                hidden_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate
            ) for _ in range(num_blocks)
        ]

        # Output network
        output_layers = []
        for _ in range(num_fc_layers):
            output_layers.extend([
                layers.Dense(hidden_dim),
                layers.BatchNormalization() if batch_norm else layers.Lambda(lambda x: x),
                layers.Activation(activation),
                layers.Dropout(dropout_rate)
            ])
        output_layers.append(layers.Dense(output_dim))
        self.output_network = keras.Sequential(output_layers)

        self.pool_method = pool_method
        self.optimizer = keras.optimizers.Adam(learning_rate)

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
            inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Forward pass computation.

        Args:
            inputs: Tuple of (node_features, edge_features, edge_indices,
                   graph_indices, global_features)
            training: Whether in training mode

        Returns:
            Graph-level predictions
        """
        node_features, edge_features, edge_indices, graph_indices, global_features = inputs

        # Initial embeddings
        node_hidden = self.node_embedding(node_features, training=training)
        edge_hidden = self.edge_embedding(edge_features, training=training)
        global_hidden = self.global_embedding(global_features, training=training)

        # Apply MEGNet blocks with residual connections
        for block in self.meg_blocks:
            node_out, edge_out, global_out = block(
                (node_hidden, edge_hidden, edge_indices, graph_indices, global_hidden),
                training=training
            )
            node_hidden = node_hidden + node_out
            edge_hidden = edge_hidden + edge_out
            global_hidden = global_hidden + global_out

        # Pool node and edge features
        node_graph = self.pool_nodes(node_hidden, graph_indices)
        edge_graph = tf.math.segment_mean(
            edge_hidden,
            tf.gather(graph_indices, edge_indices[:, 0])
        )

        # Combine all graph-level representations
        graph_features = tf.concat([node_graph, edge_graph, global_hidden], axis=-1)

        # Final prediction
        return self.output_network(graph_features, training=training)

    def train_step(self, data):
        """Custom training step with gradient clipping."""
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        # Clip gradients
        grads = tape.gradient(loss, self.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        """Returns model configuration."""
        return {
            "hidden_dim": self.meg_blocks[0].units,
            "num_blocks": len(self.meg_blocks),
            "pool_method": self.pool_method
        }