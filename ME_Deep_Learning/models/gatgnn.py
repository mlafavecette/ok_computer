"""
Graph Attention Neural Network (GATGNN) Implementation in TensorFlow

A sophisticated graph attention network specifically designed for materials science,
featuring edge feature integration, multi-head attention, and global interaction
mechanisms.

Features:
- Multi-head graph attention
- Edge feature integration
- Global attention pooling
- Residual connections
- Advanced normalization

Author: Claude
Date: 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod


class BaseAttentionLayer(layers.Layer, ABC):
    """
    Abstract base class for attention mechanisms.

    Provides common functionality for different types of attention layers
    used in the GATGNN architecture.
    """

    @abstractmethod
    def compute_attention(self, q, k, v, mask=None):
        """Compute attention weights and apply them to values."""
        pass


class EdgeAttention(BaseAttentionLayer):
    """
    Edge-aware attention mechanism.

    Computes attention scores incorporating edge features for better
    structural awareness.

    Args:
        units: Number of output features
        num_heads: Number of attention heads
        concat_heads: Whether to concatenate or average attention heads
        attention_dropout: Dropout rate for attention weights
        edge_dim: Dimension of edge features
        use_bias: Whether to use bias terms
    """

    def __init__(
            self,
            units: int,
            num_heads: int = 4,
            concat_heads: bool = True,
            attention_dropout: float = 0.0,
            edge_dim: Optional[int] = None,
            use_bias: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.head_dim = units // num_heads if concat_heads else units

        # Query, key, value projections
        self.q_proj = layers.Dense(num_heads * self.head_dim, use_bias=use_bias)
        self.k_proj = layers.Dense(num_heads * self.head_dim, use_bias=use_bias)
        self.v_proj = layers.Dense(num_heads * self.head_dim, use_bias=use_bias)

        # Edge feature projection if provided
        if edge_dim is not None:
            self.edge_proj = layers.Dense(num_heads * self.head_dim, use_bias=use_bias)
        else:
            self.edge_proj = None

        self.attention_dropout = layers.Dropout(attention_dropout)

    def compute_attention(self, q, k, v, edge_features=None, mask=None):
        """
        Compute multi-head attention with edge features.

        Args:
            q: Query tensor [batch_size, num_nodes, dim]
            k: Key tensor [batch_size, num_nodes, dim]
            v: Value tensor [batch_size, num_nodes, dim]
            edge_features: Optional edge features
            mask: Optional attention mask

        Returns:
            Updated node features
        """
        batch_size = tf.shape(q)[0]

        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])

        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True)

        # Add edge feature contributions if present
        if edge_features is not None and self.edge_proj is not None:
            edge_hidden = self.edge_proj(edge_features)
            edge_hidden = tf.reshape(edge_hidden, [-1, self.num_heads, self.head_dim])
            edge_scores = tf.einsum('bhf,bhf->bh', q, edge_hidden)
            scores = scores + edge_scores[..., None]

        # Scale scores
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        # Apply mask if provided
        if mask is not None:
            scores += (mask * -1e9)

        # Apply attention dropout
        attention = tf.nn.softmax(scores, axis=-1)
        attention = self.attention_dropout(attention)

        # Apply attention to values
        output = tf.matmul(attention, v)

        # Combine heads
        if self.concat_heads:
            output = tf.reshape(output, [batch_size, -1, self.num_heads * self.head_dim])
        else:
            output = tf.reduce_mean(output, axis=2)

        return output


class GlobalAttention(BaseAttentionLayer):
    """
    Global attention mechanism for graph-level features.

    Computes attention between nodes and global graph features for
    better graph-level representation learning.

    Args:
        units: Number of output features
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
        use_bias: Whether to use bias terms
    """

    def __init__(
            self,
            units: int,
            num_heads: int = 4,
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads

        # Projections for nodes and global features
        self.node_projection = layers.Dense(units, use_bias=use_bias)
        self.global_projection = layers.Dense(units, use_bias=use_bias)

        # Attention computation layers
        self.attention_layer = layers.Dense(num_heads, use_bias=use_bias)
        self.dropout = layers.Dropout(dropout_rate)

    def compute_attention(self, node_features, global_features, graph_indices):
        """
        Compute attention between nodes and global features.

        Args:
            node_features: Node-level features
            global_features: Global graph features
            graph_indices: Graph assignment for each node

        Returns:
            Attention weights for each node
        """
        # Project features
        node_hidden = self.node_projection(node_features)
        global_hidden = self.global_projection(global_features)

        # Gather relevant global features for each node
        global_for_nodes = tf.gather(global_hidden, graph_indices)

        # Compute attention scores
        attention_input = tf.concat([node_hidden, global_for_nodes], axis=-1)
        scores = self.attention_layer(attention_input)

        # Normalize scores within each graph
        attention = tf.math.segment_softmax(scores, graph_indices)
        attention = self.dropout(attention)

        return attention


class GATGNN(keras.Model):
    """
    Complete Graph Attention Neural Network implementation.

    Combines edge-aware attention, global attention, and sophisticated
    pooling mechanisms for materials property prediction.

    Args:
        num_features: Number of input node features
        num_edge_features: Number of edge features
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_heads: Number of attention heads
        num_layers: Number of attention layers
        dropout_rate: Dropout rate
        batch_norm: Whether to use batch normalization
        residual: Whether to use residual connections
        pool_method: Graph pooling method
        activation: Activation function
    """

    def __init__(
            self,
            num_features: int,
            num_edge_features: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_heads: int = 4,
            num_layers: int = 5,
            dropout_rate: float = 0.0,
            batch_norm: bool = True,
            residual: bool = True,
            pool_method: str = "sum",
            activation: str = "relu",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.pool_method = pool_method
        self.residual = residual

        # Initial feature projections
        self.node_embed = layers.Dense(hidden_dim)
        self.edge_embed = layers.Dense(hidden_dim)

        # Edge attention layers
        self.attention_layers = []
        for _ in range(num_layers):
            attention = EdgeAttention(
                hidden_dim,
                num_heads=num_heads,
                attention_dropout=dropout_rate,
                edge_dim=hidden_dim
            )
            self.attention_layers.append(attention)

            if batch_norm:
                self.attention_layers.append(layers.BatchNormalization())

            self.attention_layers.append(layers.Activation(activation))

            if dropout_rate > 0:
                self.attention_layers.append(layers.Dropout(dropout_rate))

        # Global attention mechanism
        self.global_attention = GlobalAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

        # Output transformation
        self.output_net = keras.Sequential([
            layers.Dense(hidden_dim, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(output_dim)
        ])

    def pool_nodes(self, node_features, attention_weights, graph_indices):
        """
        Pools node features to graph level using attention weights.

        Args:
            node_features: Node-level features
            attention_weights: Attention weights for each node
            graph_indices: Graph assignment for each node

        Returns:
            Graph-level features
        """
        # Apply attention weights
        weighted_features = node_features * attention_weights

        # Pool according to specified method
        if self.pool_method == "sum":
            return tf.math.segment_sum(weighted_features, graph_indices)
        elif self.pool_method == "mean":
            return tf.math.segment_mean(weighted_features, graph_indices)
        elif self.pool_method == "max":
            return tf.math.segment_max(weighted_features, graph_indices)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool_method}")

    def call(self, inputs, training=None):
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

        # Initial projections
        x = self.node_embed(node_features)
        edge_hidden = self.edge_embed(edge_features)

        # Store for residual connection
        if self.residual:
            residual = x

        # Apply attention layers
        for layer in self.attention_layers:
            if isinstance(layer, EdgeAttention):
                out = layer.compute_attention(
                    x, x, x,
                    edge_features=edge_hidden
                )
                if self.residual:
                    x = out + residual
                    residual = x
                else:
                    x = out
            else:
                x = layer(x, training=training)

        # Apply global attention
        attention_weights = self.global_attention.compute_attention(
            x, global_features, graph_indices
        )

        # Pool to graph level
        graph_features = self.pool_nodes(x, attention_weights, graph_indices)

        # Final prediction
        return self.output_net(graph_features)

    def get_config(self):
        """Returns model configuration."""
        return {
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "pool_method": self.pool_method,
            "residual": self.residual
        }

    @classmethod
    def from_config(cls, config):
        """Creates model from configuration."""
        return cls(**config)