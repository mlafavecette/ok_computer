import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class DiffGroupNorm(layers.Layer):
    """Differentiable Group Normalization layer.

    This layer implements group normalization with learnable parameters and
    handles different batch sizes efficiently.
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

        # Ensure number of channels is divisible by number of groups
        assert channels % num_groups == 0, (
            f"Number of channels ({channels}) must be divisible "
            f"by number of groups ({num_groups})"
        )

        # Trainable parameters
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

        # Running statistics
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

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply group normalization.

        Args:
            inputs: Input tensor [batch_size, channels] or [num_nodes, channels]
            training: Whether in training mode

        Returns:
            Normalized tensor of same shape as input
        """
        # Reshape for group normalization
        shape = tf.shape(inputs)
        batch_size = shape[0]
        num_channels = shape[-1]

        x = tf.reshape(inputs, [batch_size, self.num_groups, -1])

        if training or not self.track_running_stats:
            # Calculate statistics
            mean = tf.reduce_mean(x, axis=2)
            var = tf.reduce_variance(x, axis=2)

            if self.track_running_stats:
                # Update running statistics
                self.running_mean.assign(
                    self.running_mean * (1 - self.momentum) +
                    tf.reduce_mean(mean, axis=0) * self.momentum
                )
                self.running_var.assign(
                    self.running_var * (1 - self.momentum) +
                    tf.reduce_mean(var, axis=0) * self.momentum
                )
        else:
            # Use running statistics
            mean = tf.expand_dims(self.running_mean, 0)
            var = tf.expand_dims(self.running_var, 0)

        # Normalize
        x = (x - tf.expand_dims(mean, 2)) / tf.sqrt(tf.expand_dims(var + self.epsilon, 2))

        # Reshape back and apply affine transformation
        x = tf.reshape(x, shape)
        return x * self.gamma + self.beta


class DeepAGATConvolution(layers.Layer):
    """Deep Atomistic Graph Attention Convolution layer with skip connections.

    This layer implements an improved version of AGAT convolution with
    residual connections and enhanced attention mechanisms.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout_rate: float = 0.0,
                 use_group_norm: bool = True,
                 **kwargs):
        super(DeepAGATConvolution, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Trainable parameters for attention
        self.W = self.add_weight(
            shape=(hidden_dim * 2, num_heads * hidden_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weight'
        )

        self.att = self.add_weight(
            shape=(1, num_heads, 2 * hidden_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='attention'
        )

        self.bias = self.add_weight(
            shape=(hidden_dim,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

        # Normalization layers
        self.norm = (DiffGroupNorm(num_heads, 10) if use_group_norm
                     else layers.BatchNormalization())

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced attention mechanism.

        Args:
            inputs: Tuple of (x, edge_index, edge_attr)
            training: Whether in training mode

        Returns:
            Updated node features
        """
        x, edge_index, edge_attr = inputs

        row, col = edge_index[0], edge_index[1]

        # Gather features
        x_i = tf.gather(x, row)
        x_j = tf.gather(x, col)

        # Combine with edge features
        x_i = tf.concat([x_i, edge_attr], axis=-1)
        x_j = tf.concat([x_j, edge_attr], axis=-1)

        # Transform and reshape for multi-head attention
        x_i = tf.nn.leaky_relu(tf.matmul(x_i, self.W))
        x_j = tf.nn.leaky_relu(tf.matmul(x_j, self.W))
        x_i = tf.reshape(x_i, [-1, self.num_heads, self.hidden_dim])
        x_j = tf.reshape(x_j, [-1, self.num_heads, self.hidden_dim])

        # Compute attention scores
        alpha = tf.concat([x_i, x_j], axis=-1) * self.att
        alpha = tf.reduce_sum(alpha, axis=-1)
        alpha = tf.nn.leaky_relu(alpha, alpha=0.2)

        # Normalize attention scores
        alpha = self.norm(alpha, training=training)
        alpha = tf.nn.softmax(alpha, axis=1)  # Normalize across heads

        if training:
            alpha = tf.nn.dropout(alpha, self.dropout_rate)

        # Apply attention and aggregate
        out = x_j * tf.expand_dims(alpha, -1)
        out = tf.transpose(out, [1, 0, 2])

        # Aggregate messages
        out = tf.scatter_nd(
            tf.expand_dims(row, 1),
            tf.reshape(out, [-1, self.num_heads * self.hidden_dim]),
            shape=[tf.shape(x)[0], self.num_heads * self.hidden_dim]
        )

        # Process output
        out = tf.reshape(out, [-1, self.num_heads, self.hidden_dim])
        out = tf.reduce_mean(out, axis=1)
        out = out + self.bias

        return out


class DeepGATGNN(Model):
    """Deep Graph Attention Neural Network for crystal property prediction.

    This model implements an enhanced version of GATGNN with improved
    attention mechanisms, skip connections, and normalization schemes.
    """

    def __init__(self,
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 64,
                 num_conv_layers: int = 5,
                 num_heads: int = 4,
                 dropout_rate: float = 0.0,
                 global_features_dim: int = 108,
                 output_dim: int = 1,
                 use_group_norm: bool = True,
                 **kwargs):
        super(DeepGATGNN, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initial embeddings with skip connection preparation
        self.node_embedding = Sequential([
            layers.Dense(hidden_dim),
            layers.LeakyReLU(0.2)
        ])

        self.edge_embedding = Sequential([
            layers.Dense(hidden_dim),
            layers.LeakyReLU(0.2)
        ])

        # Deep AGAT convolution layers
        self.conv_layers = [
            DeepAGATConvolution(
                hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                use_group_norm=use_group_norm
            ) for _ in range(num_conv_layers)
        ]

        # Global attention mechanism
        self.global_attention = GlobalAttentionLayer(
            hidden_dim,
            dropout_rate=dropout_rate
        )

        # Output processing
        self.output_network = Sequential([
            layers.Dense(hidden_dim),
            layers.ReLU(),
            layers.Dropout(dropout_rate),
            layers.Dense(output_dim)
        ])

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced feature processing.

        Args:
            inputs: Tuple of (x, edge_index, edge_attr, batch_idx, global_features)
            training: Whether in training mode

        Returns:
            Predicted properties
        """
        x, edge_index, edge_attr, batch_idx, global_features = inputs

        # Initial embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        # Store initial features for residual connection
        initial_x = x

        # Apply graph convolutions with residual connections
        for i, conv in enumerate(self.conv_layers):
            x_new = conv((x, edge_index, edge_attr), training=training)
            if i > 0:  # Skip connection after first layer
                x = x + x_new
            else:
                x = x_new

        # Add global residual connection
        x = x + initial_x

        # Apply global attention
        attention_weights = self.global_attention(
            (x, batch_idx, global_features),
            training=training
        )
        x = x * attention_weights

        # Pool to graph level
        num_graphs = tf.reduce_max(batch_idx) + 1
        out = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            x,
            shape=[num_graphs, self.hidden_dim]
        )

        # Final prediction
        out = self.output_network(out, training=training)

        return out if self.output_dim > 1 else tf.squeeze(out, -1)

    def train_step(self, data):
        """Custom training step with gradient clipping and monitoring.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dict of metrics
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        # Compute and apply gradients with enhanced clipping
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        metrics['gradient_norm'] = grad_norm

        return metrics