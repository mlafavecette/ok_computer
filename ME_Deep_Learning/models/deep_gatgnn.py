import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class GlobalAttentionModule(layers.Layer):
    """
    Global Interaction Module (GIM) with attention mechanism.

    Computes attention weights based on both local node features and global context.

    Args:
        units: Number of hidden units
        activation: Activation function
        num_fc_layers: Number of fully connected layers
        dropout_rate: Dropout rate
        batch_norm: Whether to use batch normalization
        groups: Number of groups for group normalization
    """

    def __init__(
            self,
            units: int,
            activation: str = "relu",
            num_fc_layers: int = 2,
            dropout_rate: float = 0.0,
            batch_norm: bool = True,
            groups: int = 8,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.num_fc_layers = num_fc_layers
        self.dropout_rate = dropout_rate

        # Multi-layer perceptron for attention computation
        self.mlp_layers = []
        for i in range(num_fc_layers + 1):
            if i == 0:
                self.mlp_layers.append(layers.Dense(units))
            else:
                self.mlp_layers.append(layers.Dense(units if i != num_fc_layers else 1))

            if batch_norm and i != num_fc_layers:
                self.mlp_layers.append(layers.GroupNormalization(groups))

            if i != num_fc_layers:
                self.mlp_layers.append(layers.Activation(activation))
                if dropout_rate > 0:
                    self.mlp_layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs, graph_indices, global_features, training=None):
        """
        Forward pass computation.

        Args:
            inputs: Node features [num_nodes, feature_dim]
            graph_indices: Graph assignment for each node
            global_features: Global graph features [num_graphs, global_dim]
            training: Whether in training mode

        Returns:
            Attention weights for each node [num_nodes, 1]
        """
        # Combine local and global features
        global_repeated = tf.gather(global_features, graph_indices)
        combined = tf.concat([inputs, global_repeated], axis=-1)

        # Pass through MLP layers
        x = combined
        for layer in self.mlp_layers:
            x = layer(x, training=training)

        # Compute softmax over nodes in each graph
        attention = tf.math.segment_softmax(tf.squeeze(x, -1), graph_indices)

        return tf.expand_dims(attention, -1)


class MultiHeadAttentionLayer(layers.Layer):
    """
    Multi-head graph attention layer with edge features.

    Implements the core attention mechanism described in the GATGNN paper.

    Args:
        units: Number of output features
        num_heads: Number of attention heads
        edge_dim: Dimension of edge features
        activation: Activation function
        dropout_rate: Dropout rate
        use_bias: Whether to use bias terms
        kernel_initializer: Weight initialization method
    """

    def __init__(
            self,
            units: int,
            num_heads: int = 4,
            edge_dim: int = None,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: str = "glorot_uniform",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Linear transformations for queries, keys, values
        self.query_transform = layers.Dense(
            units * num_heads,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )
        self.key_transform = layers.Dense(
            units * num_heads,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )
        self.value_transform = layers.Dense(
            units * num_heads,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

        # Edge feature transform if edge features present
        if edge_dim is not None:
            self.edge_transform = layers.Dense(
                units * num_heads,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer
            )
        else:
            self.edge_transform = None

        # Output transformation
        self.output_transform = layers.Dense(
            units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

        self.dropout = layers.Dropout(dropout_rate)
        self.activation_fn = layers.Activation(activation)

    def call(self, inputs, edge_index, edge_features=None, training=None):
        """
        Forward pass computation.

        Args:
            inputs: Node features [num_nodes, feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Optional edge features [num_edges, edge_dim]
            training: Whether in training mode

        Returns:
            Updated node features
        """
        num_nodes = tf.shape(inputs)[0]

        # Compute queries, keys, values
        q = self.query_transform(inputs)
        k = self.key_transform(inputs)
        v = self.value_transform(inputs)

        # Reshape to [num_nodes, num_heads, units]
        q = tf.reshape(q, [-1, self.num_heads, self.units])
        k = tf.reshape(k, [-1, self.num_heads, self.units])
        v = tf.reshape(v, [-1, self.num_heads, self.units])

        # Gather connected nodes
        node_i = tf.gather(q, edge_index[0])
        node_j = tf.gather(k, edge_index[1])

        # Compute attention scores
        scores = tf.reduce_sum(node_i * node_j, axis=-1)

        # Include edge features if present
        if edge_features is not None and self.edge_transform is not None:
            edge_hidden = self.edge_transform(edge_features)
            edge_hidden = tf.reshape(edge_hidden, [-1, self.num_heads, self.units])
            scores = scores + tf.reduce_sum(edge_hidden * node_j, axis=-1)

        # Normalize attention scores
        attention = tf.math.segment_softmax(scores, edge_index[0])
        attention = self.dropout(attention, training=training)

        # Apply attention to values
        values = tf.gather(v, edge_index[1])
        attended = tf.math.segment_sum(
            tf.expand_dims(attention, -1) * values,
            edge_index[0]
        )

        # Combine heads and transform output
        attended = tf.reshape(attended, [-1, self.num_heads * self.units])
        outputs = self.output_transform(attended)
        outputs = self.activation_fn(outputs)

        return outputs


class DeepGATGNN(keras.Model):
    """
    Complete Deep Graph Attention Neural Network architecture.

    Combines multiple attention layers with global attention and
    sophisticated pooling mechanisms.

    Args:
        num_features: Number of input node features
        num_edge_features: Number of edge features
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_heads: Number of attention heads
        num_layers: Number of attention layers
        global_layers: Number of global attention layers
        dropout_rate: Dropout rate
        batch_norm: Whether to use batch normalization
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
            global_layers: int = 2,
            dropout_rate: float = 0.0,
            batch_norm: bool = True,
            pool_method: str = "mean",
            activation: str = "relu",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Initial transforms for node and edge features
        self.node_embed = layers.Dense(hidden_dim)
        self.edge_embed = layers.Dense(hidden_dim)

        # Graph attention layers
        self.attention_layers = []
        for _ in range(num_layers):
            self.attention_layers.append(
                MultiHeadAttentionLayer(
                    hidden_dim,
                    num_heads=num_heads,
                    edge_dim=hidden_dim,
                    activation=activation,
                    dropout_rate=dropout_rate
                )
            )
            if batch_norm:
                self.attention_layers.append(layers.BatchNormalization())

        # Global attention module
        self.global_attention = GlobalAttentionModule(
            hidden_dim,
            activation=activation,
            num_fc_layers=global_layers,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm
        )

        # Output layers
        self.output_layers = [
            layers.Dense(hidden_dim, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(output_dim)
        ]

        self.pool_method = pool_method

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

        # Initial feature transforms
        x = self.node_embed(node_features)
        edge_hidden = self.edge_embed(edge_features)

        # Store residual
        residual = x

        # Apply attention layers
        for layer in self.attention_layers:
            if isinstance(layer, MultiHeadAttentionLayer):
                x = layer(x, edge_indices, edge_hidden, training=training)
                # Add residual connection
                x = x + residual
                residual = x
            else:
                x = layer(x, training=training)

        # Apply global attention
        attention_weights = self.global_attention(
            x,
            graph_indices,
            global_features,
            training=training
        )
        x = x * attention_weights

        # Pool node features to graph level
        if self.pool_method == "mean":
            x = tf.math.segment_mean(x, graph_indices)
        elif self.pool_method == "sum":
            x = tf.math.segment_sum(x, graph_indices)
        elif self.pool_method == "max":
            x = tf.math.segment_max(x, graph_indices)

        # Final output transformation
        for layer in self.output_layers:
            x = layer(x, training=training)

        return x

    def get_config(self):
        """Returns model configuration."""
        return {
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "pool_method": self.pool_method
        }