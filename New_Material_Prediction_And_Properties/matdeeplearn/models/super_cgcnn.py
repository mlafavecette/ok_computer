import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class DiffGroupNorm(layers.Layer):
    """Differentiable Group Normalization layer optimized for crystal graphs.

    This layer implements group normalization with learnable parameters and
    gradient-friendly operations, specifically designed for crystal structures.
    """

    def __init__(self,
                 channels: int,
                 num_groups: int = 10,
                 epsilon: float = 1e-5,
                 track_running_stats: bool = True,
                 **kwargs):
        super(DiffGroupNorm, self).__init__(**kwargs)

        self.channels = channels
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.track_running_stats = track_running_stats

        assert channels % num_groups == 0, "Channels must be divisible by number of groups"

        # Learnable parameters
        self.gamma = self.add_weight(
            shape=(channels,),
            initializer='ones',
            trainable=True,
            name='gamma'
        )
        self.beta = self.add_weight(
            shape=(channels,),
            initializer='zeros',
            trainable=True,
            name='beta'
        )

        if track_running_stats:
            self.running_mean = self.add_weight(
                shape=(num_groups,),
                initializer='zeros',
                trainable=False,
                name='running_mean'
            )
            self.running_var = self.add_weight(
                shape=(num_groups,),
                initializer='ones',
                trainable=False,
                name='running_var'
            )
            self.momentum = 0.1

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply group normalization with running statistics.

        Args:
            inputs: Input features [batch_size or num_nodes, channels]
            training: Whether in training mode

        Returns:
            Normalized features of same shape as input
        """
        # Reshape for group normalization
        shape = tf.shape(inputs)
        batch_size = shape[0]
        channels_per_group = self.channels // self.num_groups

        x = tf.reshape(inputs, [batch_size, self.num_groups, channels_per_group])

        if training or not self.track_running_stats:
            mean = tf.reduce_mean(x, axis=[0, 2])
            var = tf.reduce_variance(x, axis=[0, 2])

            if self.track_running_stats:
                # Update running statistics
                self.running_mean.assign(
                    self.running_mean * (1 - self.momentum) +
                    mean * self.momentum
                )
                self.running_var.assign(
                    self.running_var * (1 - self.momentum) +
                    var * self.momentum
                )
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x = (x - tf.reshape(mean, [1, -1, 1])) / \
            tf.sqrt(tf.reshape(var + self.epsilon, [1, -1, 1]))

        # Reshape and apply learnable parameters
        x = tf.reshape(x, shape)
        return x * self.gamma + self.beta


class SuperCGConv(layers.Layer):
    """Enhanced Crystal Graph Convolutional layer.

    This layer implements an improved version of crystal graph convolutions
    with enhanced feature processing and residual connections.
    """

    def __init__(self,
                 hidden_dim: int,
                 edge_dim: int,
                 activation: str = 'relu',
                 use_group_norm: bool = True,
                 **kwargs):
        super(SuperCGConv, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # Edge network
        self.edge_network = Sequential([
            layers.Dense(2 * hidden_dim, activation=activation),
            layers.Dense(hidden_dim)
        ])

        # Node update network
        self.node_network = Sequential([
            layers.Dense(hidden_dim, activation=activation),
            layers.Dense(hidden_dim)
        ])

        # Normalization
        if use_group_norm:
            self.norm = DiffGroupNorm(hidden_dim, 10)
        else:
            self.norm = layers.BatchNormalization()

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced feature processing.

        Args:
            inputs: Tuple of (x, edge_index, edge_attr)
            training: Whether in training mode

        Returns:
            Updated node features
        """
        x, edge_index, edge_attr = inputs
        row, col = edge_index[0], edge_index[1]

        # Edge feature processing
        edge_features = self.edge_network(edge_attr)

        # Gather node features
        x_j = tf.gather(x, col)
        x_i = tf.gather(x, row)

        # Combine node and edge features
        messages = x_j * edge_features

        # Aggregate messages
        out = tf.scatter_nd(
            tf.expand_dims(row, 1),
            messages,
            shape=[tf.shape(x)[0], self.hidden_dim]
        )

        # Update node features
        out = self.node_network(out)
        out = self.norm(out, training=training)

        return out


class SuperCGCNN(Model):
    """Enhanced Crystal Graph Convolutional Neural Network.

    This model implements an improved version of CGCNN with advanced features:
    - Enhanced convolution operations
    - Sophisticated residual connections
    - Improved normalization scheme
    - Better gradient flow
    - Advanced pooling strategies
    """

    def __init__(self,
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 64,
                 num_conv_layers: int = 3,
                 num_dense_layers: int = 2,
                 output_dim: int = 1,
                 dropout_rate: float = 0.0,
                 use_group_norm: bool = True,
                 **kwargs):
        super(SuperCGCNN, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initial embeddings
        self.node_embedding = Sequential([
            layers.Dense(hidden_dim),
            layers.ReLU()
        ])

        # Convolution layers
        self.conv_layers = []
        self.norms = []

        for _ in range(num_conv_layers):
            self.conv_layers.append(
                SuperCGConv(hidden_dim, edge_features, use_group_norm=use_group_norm)
            )
            if use_group_norm:
                self.norms.append(DiffGroupNorm(hidden_dim, 10))
            else:
                self.norms.append(layers.BatchNormalization())

        # Output network
        output_layers = []
        for _ in range(num_dense_layers):
            output_layers.extend([
                layers.Dense(hidden_dim, activation='relu'),
                layers.Dropout(dropout_rate)
            ])
        output_layers.append(layers.Dense(output_dim))
        self.output_network = Sequential(output_layers)

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced residual connections.

        Args:
            inputs: Tuple of (node_attr, edge_index, edge_attr, batch_idx)
            training: Whether in training mode

        Returns:
            Predicted properties
        """
        x, edge_index, edge_attr, batch_idx = inputs

        # Initial embedding
        x = self.node_embedding(x)

        # Store initial features for global residual
        initial_features = x

        # Convolution layers with residual connections
        for conv, norm in zip(self.conv_layers, self.norms):
            x_conv = conv((x, edge_index, edge_attr), training=training)
            x_conv = norm(x_conv, training=training)

            # Local residual connection
            x = x + x_conv

        # Global residual connection
        x = x + initial_features

        # Pool nodes to graph level
        num_graphs = tf.reduce_max(batch_idx) + 1
        pooled = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            x,
            shape=[num_graphs, self.hidden_dim]
        )

        # Final prediction
        out = self.output_network(pooled, training=training)
        return out if self.output_dim > 1 else tf.squeeze(out, -1)

    def train_step(self, data):
        """Custom training step with gradient monitoring and clipping.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dict of metrics
        """
        inputs, targets = data

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(targets, predictions)

        # Compute and clip gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(targets, predictions)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({
            'loss': loss,
            'gradient_norm': grad_norm
        })

        return metrics