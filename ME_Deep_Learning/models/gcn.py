"""
Graph Convolutional Network (GCN) Implementation in TensorFlow

A modern implementation of the Graph Convolutional Network architecture
optimized for materials science applications. Includes spectral and spatial
convolution variants with sophisticated pooling mechanisms.

Features:
- Spectral and spatial graph convolutions
- Residual connections
- Advanced normalization techniques
- Flexible pooling strategies

Author: Claude
Date: 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class GraphConvLayer(layers.Layer):
    """
    Graph Convolutional Layer implementation.

    Implements the spectral-based graph convolution operation with optional
    edge weights and self-loops.

    Args:
        units: Number of output features
        use_bias: Whether to use bias terms
        activation: Activation function
        dropout_rate: Dropout rate
        kernel_initializer: Weight initialization method
        bias_initializer: Bias initialization method
        batch_norm: Whether to use batch normalization
        improved: Whether to use improved GCN formulation
        add_self_loops: Whether to add self-loops to adjacency
    """

    def __init__(
            self,
            units: int,
            use_bias: bool = True,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            kernel_initializer: str = "glorot_uniform",
            bias_initializer: str = "zeros",
            batch_norm: bool = True,
            improved: bool = True,
            add_self_loops: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.improved = improved
        self.add_self_loops = add_self_loops

        # Core transform layers
        self.feature_transform = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=kernel_initializer
        )

        if use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=(units,),
                initializer=bias_initializer,
                trainable=True
            )

        # Regularization and normalization
        self.dropout = layers.Dropout(dropout_rate)
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        self.activation = layers.Activation(activation)

    def build(self, input_shape):
        """Builds layer weights based on input shape."""
        super().build(input_shape)

    def compute_adjacency_norm(self, edge_index, edge_weight, num_nodes):
        """
        Computes normalized adjacency matrix.

        Args:
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            num_nodes: Number of nodes in graph

        Returns:
            Normalized adjacency matrix
        """
        if edge_weight is None:
            edge_weight = tf.ones([tf.shape(edge_index)[1]])

        # Add self-loops if specified
        if self.add_self_loops:
            loop_index = tf.range(num_nodes)
            loop_index = tf.stack([loop_index, loop_index])
            edge_index = tf.concat([edge_index, loop_index], axis=1)
            loop_weight = tf.ones([num_nodes])
            edge_weight = tf.concat([edge_weight, loop_weight], axis=0)

        # Compute degree matrix
        row, col = edge_index[0], edge_index[1]
        deg = tf.math.unsorted_segment_sum(edge_weight, row, num_nodes)

        # Improved GCN normalization
        if self.improved:
            deg = tf.pow(deg, -0.5)
            deg = tf.where(tf.math.is_inf(deg), 0., deg)
            norm = tf.gather(deg, row) * edge_weight * tf.gather(deg, col)
        else:
            deg = 1.0 / deg
            deg = tf.where(tf.math.is_inf(deg), 0., deg)
            norm = tf.gather(deg, row) * edge_weight

        return edge_index, norm

    def call(self, inputs, training=None):
        """
        Forward pass computation.

        Args:
            inputs: Tuple of (node_features, edge_index, edge_weight)
            training: Whether in training mode

        Returns:
            Updated node features
        """
        x, edge_index, edge_weight = inputs
        num_nodes = tf.shape(x)[0]

        # Transform features
        x = self.feature_transform(x)

        # Compute normalized adjacency
        edge_index, norm = self.compute_adjacency_norm(
            edge_index, edge_weight, num_nodes
        )

        # Message passing
        row, col = edge_index[0], edge_index[1]
        x_j = tf.gather(x, col)
        messages = x_j * tf.expand_dims(norm, -1)
        x = tf.math.unsorted_segment_sum(messages, row, num_nodes)

        # Apply bias if specified
        if self.use_bias:
            x = x + self.bias

        # Apply normalization and regularization
        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.activation(x)

        return x

    def get_config(self):
        """Returns layer configuration."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "improved": self.improved,
            "add_self_loops": self.add_self_loops
        })
        return config


class GCN(keras.Model):
    """
    Complete Graph Convolutional Network architecture.

    A flexible implementation that allows for different variants of the
    architecture through configuration options.

    Args:
        num_features: Number of input node features
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of GCN layers
        dropout_rate: Dropout rate
        batch_norm: Whether to use batch normalization
        improved: Whether to use improved GCN formulation
        residual: Whether to use residual connections
        pool_method: Graph pooling method
        activation: Activation function
    """

    def __init__(
            self,
            num_features: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_layers: int = 3,
            dropout_rate: float = 0.0,
            batch_norm: bool = True,
            improved: bool = True,
            residual: bool = True,
            pool_method: str = "mean",
            activation: str = "relu",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.pool_method = pool_method

        # Input projection
        self.input_project = layers.Dense(hidden_dim)

        # GCN layers
        self.conv_layers = []
        for _ in range(num_layers):
            self.conv_layers.append(
                GraphConvLayer(
                    hidden_dim,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                    improved=improved,
                    activation=activation
                )
            )

        # Output layers
        self.output_transform = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(output_dim)
        ])

        self.residual = residual

    def pool_nodes(self, node_features, graph_indices):
        """
        Pools node features to graph level.

        Args:
            node_features: Node-level features [num_nodes, feature_dim]
            graph_indices: Graph assignment for each node

        Returns:
            Graph-level features [num_graphs, feature_dim]
        """
        if self.pool_method == "mean":
            return tf.math.segment_mean(node_features, graph_indices)
        elif self.pool_method == "sum":
            return tf.math.segment_sum(node_features, graph_indices)
        elif self.pool_method == "max":
            return tf.math.segment_max(node_features, graph_indices)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool_method}")

    def call(self, inputs, training=None):
        """
        Forward pass computation.

        Args:
            inputs: Tuple of (node_features, edge_index, edge_weight, graph_indices)
            training: Whether in training mode

        Returns:
            Graph-level predictions
        """
        x, edge_index, edge_weight, graph_indices = inputs

        # Initial projection
        x = self.input_project(x)

        # Store initial features for residual
        if self.residual:
            residual = x

        # Apply GCN layers
        for conv in self.conv_layers:
            out = conv((x, edge_index, edge_weight), training=training)
            if self.residual:
                x = out + residual
                residual = x
            else:
                x = out

        # Pool to graph level
        x = self.pool_nodes(x, graph_indices)

        # Final prediction
        return self.output_transform(x)

    def get_config(self):
        """Returns model configuration."""
        return {
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "pool_method": self.pool_method,
            "residual": self.residual
        }