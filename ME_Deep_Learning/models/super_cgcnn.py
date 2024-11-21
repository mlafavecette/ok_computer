"""
Super Crystal Graph Convolutional Neural Network (SuperCGCNN) Implementation

An enhanced version of CGCNN with advanced architectural features for materials
property prediction. Incorporates sophisticated normalization, dynamic residual
connections, and multi-scale feature processing.

Features:
- Dynamic group normalization
- Advanced residual connections
- Multi-scale feature aggregation
- Adaptive feature fusion
- Enhanced gradient flow

Author: Claude
Date: 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class DynamicGroupNorm(layers.Layer):
    """
    Dynamic Group Normalization layer.

    Implements group normalization with dynamic group size adjustment
    based on input statistics.

    Args:
        units: Number of features
        min_groups: Minimum number of groups
        max_groups: Maximum number of groups
        epsilon: Small constant for numerical stability
        momentum: Momentum for running statistics
    """

    def __init__(
            self,
            units: int,
            min_groups: int = 4,
            max_groups: int = 32,
            epsilon: float = 1e-5,
            momentum: float = 0.99,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.min_groups = min_groups
        self.max_groups = max_groups
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = self.add_weight(
            "gamma",
            shape=(units,),
            initializer="ones",
            trainable=True
        )
        self.beta = self.add_weight(
            "beta",
            shape=(units,),
            initializer="zeros",
            trainable=True
        )

    def compute_groups(self, inputs: tf.Tensor) -> int:
        """Dynamically compute optimal number of groups."""
        # Compute feature correlations
        corr = tf.abs(tf.correlation(inputs, inputs))
        mean_corr = tf.reduce_mean(corr)

        # Adjust groups based on correlation
        groups = tf.cast(
            tf.clip_by_value(
                self.min_groups + (1 - mean_corr) * (self.max_groups - self.min_groups),
                self.min_groups,
                self.max_groups
            ),
            tf.int32
        )
        return groups

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply dynamic group normalization."""
        groups = self.compute_groups(inputs)

        # Reshape for group normalization
        shape = tf.shape(inputs)
        group_shape = [shape[0], groups, self.units // groups]
        x = tf.reshape(inputs, group_shape)

        # Compute statistics
        mean, variance = tf.nn.moments(x, axes=[2], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + self.epsilon)

        # Reshape back
        x = tf.reshape(x, shape)

        return x * self.gamma + self.beta


class EnhancedCGConv(layers.Layer):
    """
    Enhanced Crystal Graph Convolution layer.

    Implements advanced message passing with multi-scale feature processing
    and adaptive feature fusion.

    Args:
        units: Number of output features
        edge_units: Number of edge features
        activation: Activation function
        normalization: Normalization method
        aggregation: Message aggregation method
        num_kernels: Number of parallel kernels
    """

    def __init__(
            self,
            units: int,
            edge_units: int,
            activation: str = "relu",
            normalization: str = "dynamic_group",
            aggregation: str = "mean",
            num_kernels: int = 3,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.edge_units = edge_units
        self.num_kernels = num_kernels

        # Multi-scale kernels
        self.kernels = [
            layers.Dense(units, use_bias=False)
            for _ in range(num_kernels)
        ]

        # Edge network
        self.edge_network = keras.Sequential([
            layers.Dense(units),
            layers.Activation(activation),
            layers.Dense(units * num_kernels)
        ])

        # Feature fusion
        self.fusion = layers.Dense(units)

        # Normalization
        if normalization == "dynamic_group":
            self.norm = DynamicGroupNorm(units)
        elif normalization == "batch":
            self.norm = layers.BatchNormalization()
        else:
            self.norm = None

        self.activation = layers.Activation(activation)
        self.aggregation = aggregation

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Forward pass computation.

        Args:
            inputs: Tuple of (node_features, edge_indices, edge_features)
            training: Whether in training mode

        Returns:
            Updated node features
        """
        node_features, edge_indices, edge_features = inputs

        # Process edge features
        edge_weights = self.edge_network(edge_features)
        edge_weights = tf.reshape(edge_weights, [-1, self.num_kernels, self.units])

        # Multi-scale message passing
        messages = []
        for i in range(self.num_kernels):
            # Get source node features
            source_features = tf.gather(node_features, edge_indices[0])
            source_features = self.kernels[i](source_features)

            # Apply edge weights
            msg = source_features * edge_weights[:, i, :]

            # Aggregate messages
            if self.aggregation == "mean":
                msg = tf.math.unsorted_segment_mean(
                    msg,
                    edge_indices[1],
                    num_segments=tf.shape(node_features)[0]
                )
            elif self.aggregation == "sum":
                msg = tf.math.unsorted_segment_sum(
                    msg,
                    edge_indices[1],
                    num_segments=tf.shape(node_features)[0]
                )

            messages.append(msg)

        # Combine messages
        combined = tf.concat(messages, axis=-1)
        output = self.fusion(combined)

        # Apply normalization
        if self.norm is not None:
            output = self.norm(output, training=training)

        return self.activation(output)


class SuperCGCNN(keras.Model):
    """
    Enhanced Crystal Graph Convolutional Neural Network.

    Features advanced architectural components for improved
    materials property prediction.

    Args:
        num_features: Number of input node features
        num_edge_features: Number of edge features
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_conv_layers: Number of convolution layers
        num_dense_layers: Number of dense layers
        activation: Activation function
        normalization: Normalization method
        dropout_rate: Dropout rate
        residual: Whether to use residual connections
        pool_method: Graph pooling method
    """

    def __init__(
            self,
            num_features: int,
            num_edge_features: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_conv_layers: int = 3,
            num_dense_layers: int = 2,
            activation: str = "relu",
            normalization: str = "dynamic_group",
            dropout_rate: float = 0.0,
            residual: bool = True,
            pool_method: str = "mean",
            **kwargs
    ):
        super().__init__(**kwargs)

        # Initial embedding
        self.node_embedding = keras.Sequential([
            layers.Dense(hidden_dim),
            DynamicGroupNorm(hidden_dim) if normalization == "dynamic_group"
            else layers.BatchNormalization() if normalization == "batch"
            else layers.Lambda(lambda x: x),
            layers.Activation(activation),
            layers.Dropout(dropout_rate)
        ])

        # Convolution layers
        self.conv_layers = []
        for _ in range(num_conv_layers):
            conv = EnhancedCGConv(
                hidden_dim,
                num_edge_features,
                activation=activation,
                normalization=normalization
            )
            self.conv_layers.append(conv)

            if dropout_rate > 0:
                self.conv_layers.append(layers.Dropout(dropout_rate))

        # Output network
        self.output_network = []
        for _ in range(num_dense_layers):
            self.output_network.extend([
                layers.Dense(hidden_dim),
                DynamicGroupNorm(hidden_dim) if normalization == "dynamic_group"
                else layers.BatchNormalization() if normalization == "batch"
                else layers.Lambda(lambda x: x),
                layers.Activation(activation),
                layers.Dropout(dropout_rate)
            ])
        self.output_network.append(layers.Dense(output_dim))
        self.output_network = keras.Sequential(self.output_network)

        self.residual = residual
        self.pool_method = pool_method

    def pool_nodes(self, node_features: tf.Tensor, graph_indices: tf.Tensor) -> tf.Tensor:
        """Pool node features to graph level."""
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
            inputs: Tuple of (node_features, edge_indices, edge_features, graph_indices)
            training: Whether in training mode

        Returns:
            Graph-level predictions
        """
        node_features, edge_indices, edge_features, graph_indices = inputs

        # Initial embedding
        hidden = self.node_embedding(node_features, training=training)

        # Store for residual
        if self.residual:
            residual = hidden

        # Apply convolution layers
        for layer in self.conv_layers:
            if isinstance(layer, EnhancedCGConv):
                conv_out = layer(
                    (hidden, edge_indices, edge_features),
                    training=training
                )
                if self.residual:
                    hidden = conv_out + residual
                    residual = hidden
                else:
                    hidden = conv_out
            else:
                hidden = layer(hidden, training=training)

        # Pool to graph level
        graph_features = self.pool_nodes(hidden, graph_indices)

        # Final prediction
        return self.output_network(graph_features, training=training)

    def get_config(self):
        """Returns model configuration."""
        return {
            "hidden_dim": self.conv_layers[0].units,
            "num_conv_layers": len([l for l in self.conv_layers
                                    if isinstance(l, EnhancedCGConv)]),
            "residual": self.residual,
            "pool_method": self.pool_method
        }