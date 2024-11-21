import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class SuperMegnetEdgeModel(layers.Layer):
    """Enhanced edge update model for Super-MEGNet.

    This layer implements an improved version of edge updates with
    sophisticated residual connections and group normalization.
    """

    def __init__(self,
                 hidden_dim: int,
                 activation: str = 'softplus',
                 use_group_norm: bool = True,
                 dropout_rate: float = 0.0,
                 **kwargs):
        super(SuperMegnetEdgeModel, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Input combination network
        self.combine = Sequential([
            layers.Dense(hidden_dim),
            layers.Activation(activation)
        ])

        # Edge update networks
        self.edge_networks = []
        self.norms = []

        for _ in range(2):  # Use 2 update blocks for better feature processing
            self.edge_networks.append(Sequential([
                layers.Dense(hidden_dim),
                layers.Activation(activation),
                layers.Dense(hidden_dim)
            ]))

            if use_group_norm:
                self.norms.append(DiffGroupNorm(hidden_dim, 10))
            else:
                self.norms.append(layers.BatchNormalization())

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced residual connections.

        Args:
            inputs: Tuple of (edge_attr, src_features, dst_features, global_features)

        Returns:
            Updated edge features
        """
        edge_attr, src, dst, global_feat = inputs

        # Combine inputs
        combined = tf.concat([
            edge_attr,
            src,
            dst,
            tf.repeat(global_feat, tf.shape(edge_attr)[0] // tf.shape(global_feat)[0], axis=0)
        ], axis=-1)

        out = self.combine(combined)
        initial_features = out

        # Multi-block processing with residual connections
        for network, norm in zip(self.edge_networks, self.norms):
            block_out = network(out)
            block_out = norm(block_out, training=training)

            if training:
                block_out = tf.nn.dropout(block_out, self.dropout_rate)

            out = out + block_out

        # Global residual connection
        return out + initial_features


class SuperMegnetNodeModel(layers.Layer):
    """Enhanced node update model for Super-MEGNet.

    This layer implements improved node updates with sophisticated
    message aggregation and multi-block processing.
    """

    def __init__(self,
                 hidden_dim: int,
                 activation: str = 'softplus',
                 use_group_norm: bool = True,
                 dropout_rate: float = 0.0,
                 **kwargs):
        super(SuperMegnetNodeModel, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Input combination network
        self.combine = Sequential([
            layers.Dense(hidden_dim),
            layers.Activation(activation)
        ])

        # Node update networks
        self.node_networks = []
        self.norms = []

        for _ in range(2):
            self.node_networks.append(Sequential([
                layers.Dense(hidden_dim),
                layers.Activation(activation),
                layers.Dense(hidden_dim)
            ]))

            if use_group_norm:
                self.norms.append(DiffGroupNorm(hidden_dim, 10))
            else:
                self.norms.append(layers.BatchNormalization())

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced message aggregation.

        Args:
            inputs: Tuple of (x, edge_index, edge_features, global_features)

        Returns:
            Updated node features
        """
        x, edge_index, edge_features, global_feat = inputs
        row, col = edge_index[0], edge_index[1]

        # Aggregate messages from edges
        messages = tf.scatter_nd(
            tf.expand_dims(row, 1),
            edge_features,
            shape=[tf.shape(x)[0], self.hidden_dim]
        )

        # Combine with global features
        combined = tf.concat([
            x,
            messages,
            tf.repeat(global_feat, tf.shape(x)[0] // tf.shape(global_feat)[0], axis=0)
        ], axis=-1)

        out = self.combine(combined)
        initial_features = out

        # Multi-block processing with residual connections
        for network, norm in zip(self.node_networks, self.norms):
            block_out = network(out)
            block_out = norm(block_out, training=training)

            if training:
                block_out = tf.nn.dropout(block_out, self.dropout_rate)

            out = out + block_out

        return out + initial_features


class SuperMegnetGlobalModel(layers.Layer):
    """Enhanced global state update model for Super-MEGNet.

    This layer implements improved global state updates with sophisticated
    feature aggregation and multi-scale processing.
    """

    def __init__(self,
                 hidden_dim: int,
                 activation: str = 'softplus',
                 use_group_norm: bool = True,
                 dropout_rate: float = 0.0,
                 **kwargs):
        super(SuperMegnetGlobalModel, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Input combination network
        self.combine = Sequential([
            layers.Dense(hidden_dim),
            layers.Activation(activation)
        ])

        # Global update networks
        self.global_networks = []
        self.norms = []

        for _ in range(2):
            self.global_networks.append(Sequential([
                layers.Dense(hidden_dim),
                layers.Activation(activation),
                layers.Dense(hidden_dim)
            ]))

            if use_group_norm:
                self.norms.append(DiffGroupNorm(hidden_dim, 10))
            else:
                self.norms.append(layers.BatchNormalization())

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced feature aggregation.

        Args:
            inputs: Tuple of (global_features, node_features, edge_features, batch_idx)

        Returns:
            Updated global features
        """
        global_feat, node_features, edge_features, batch_idx = inputs

        # Aggregate node and edge features
        node_aggr = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            node_features,
            shape=[tf.shape(global_feat)[0], self.hidden_dim]
        )

        edge_aggr = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            edge_features,
            shape=[tf.shape(global_feat)[0], self.hidden_dim]
        )

        # Combine features
        combined = tf.concat([global_feat, node_aggr, edge_aggr], axis=-1)
        out = self.combine(combined)
        initial_features = out

        # Multi-block processing with residual connections
        for network, norm in zip(self.global_networks, self.norms):
            block_out = network(out)
            block_out = norm(block_out, training=training)

            if training:
                block_out = tf.nn.dropout(block_out, self.dropout_rate)

            out = out + block_out

        return out + initial_features


class SuperMEGNet(Model):
    """Super Materials Exploration Graph Neural Network.

    This model implements an enhanced version of MEGNet with:
    - Sophisticated multi-block processing
    - Advanced residual connections
    - Improved normalization schemes
    - Enhanced feature aggregation
    - Better gradient flow
    - Physics-informed architectural choices
    """

    def __init__(self,
                 node_features: int,
                 edge_features: int,
                 global_features: int,
                 hidden_dim: int = 64,
                 num_blocks: int = 3,
                 num_dense_layers: int = 2,
                 output_dim: int = 1,
                 activation: str = 'softplus',
                 dropout_rate: float = 0.0,
                 use_group_norm: bool = True,
                 **kwargs):
        super(SuperMEGNet, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initial embeddings with pre-norm
        self.node_embedding = Sequential([
            layers.Dense(hidden_dim),
            DiffGroupNorm(hidden_dim, 10) if use_group_norm else layers.BatchNormalization(),
            layers.Activation(activation)
        ])

        self.edge_embedding = Sequential([
            layers.Dense(hidden_dim),
            DiffGroupNorm(hidden_dim, 10) if use_group_norm else layers.BatchNormalization(),
            layers.Activation(activation)
        ])

        self.global_embedding = Sequential([
            layers.Dense(hidden_dim),
            DiffGroupNorm(hidden_dim, 10) if use_group_norm else layers.BatchNormalization(),
            layers.Activation(activation)
        ])

        # Multi-block message passing
        self.megnet_blocks = []
        for _ in range(num_blocks):
            self.megnet_blocks.append({
                'edge_model': SuperMegnetEdgeModel(hidden_dim, activation, use_group_norm, dropout_rate),
                'node_model': SuperMegnetNodeModel(hidden_dim, activation, use_group_norm, dropout_rate),
                'global_model': SuperMegnetGlobalModel(hidden_dim, activation, use_group_norm, dropout_rate)
            })

        # Output network with attention
        self.attention = layers.Dense(1, activation='sigmoid')

        output_layers = []
        for _ in range(num_dense_layers):
            output_layers.extend([
                layers.Dense(hidden_dim),
                layers.Activation(activation),
                layers.Dropout(dropout_rate)
            ])
        output_layers.append(layers.Dense(output_dim))
        self.output_network = Sequential(output_layers)

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced feature processing.

        Args:
            inputs: Tuple of (node_attr, edge_index, edge_attr, global_attr, batch_idx)

        Returns:
            Predicted properties
        """
        x, edge_index, edge_attr, global_attr, batch_idx = inputs

        # Initial embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        global_attr = self.global_embedding(global_attr)

        # Store initial features
        initial_x = x
        initial_edge = edge_attr
        initial_global = global_attr

        # Process through MEGNet blocks with residual connections
        for block in self.megnet_blocks:
            # Update features
            edge_out = block['edge_model'](
                (edge_attr, tf.gather(x, edge_index[0]), tf.gather(x, edge_index[1]), global_attr),
                training=training
            )

            node_out = block['node_model'](
                (x, edge_index, edge_out, global_attr),
                training=training
            )

            global_out = block['global_model'](
                (global_attr, node_out, edge_out, batch_idx),
                training=training
            )

            # Update with residual connections
            edge_attr = edge_attr + edge_out
            x = x + node_out
            global_attr = global_attr + global_out

        # Global residual connections
        x = x + initial_x
        edge_attr = edge_attr + initial_edge
        global_attr = global_attr + initial_global

        # Attention-weighted node pooling
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Pool to graph level
        num_graphs = tf.reduce_max(batch_idx) + 1
        node_pool = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            x,
            shape=[num_graphs, self.hidden_dim]
        )

        # Combine with global features
        final_features = tf.concat([node_pool, global_attr], axis=-1)

        # Final prediction
        out = self.output_network(final_features, training=training)
        return out if self.output_dim > 1 else tf.squeeze(out, -1)

    def train_step(self, data):
        """Custom training step with comprehensive monitoring.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dict of metrics
        """
        inputs, targets = data

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)

            # Compute losses
            main_loss = self.compiled_loss(targets, predictions)
            reg_loss = tf.add_n(self.losses) if self.losses else 0.0
            total_loss = main_loss + reg_loss

        # Compute and clip gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(targets, predictions)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({
            'loss': total_loss,
            'main_loss': main_loss,
            'reg_loss': reg_loss,
            'gradient_norm': grad_norm
        })

        return metrics