import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class MPNNConvolution(layers.Layer):
    """Message Passing Neural Network Convolution layer.

    This layer implements the message passing operation with edge networks and GRU updates,
    designed specifically for molecular and crystal systems.
    """

    def __init__(self,
                 hidden_dim: int,
                 edge_dim: int,
                 aggregation: str = 'mean',
                 use_bias: bool = True,
                 **kwargs):
        super(MPNNConvolution, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # Edge network
        self.edge_network = Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim * hidden_dim)
        ])

        # GRU for state updates
        self.gru = layers.GRU(hidden_dim, return_sequences=True, return_state=True)

    def build(self, input_shape):
        super(MPNNConvolution, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass of the layer.

        Args:
            inputs: Tuple of (x, edge_index, edge_attr, hidden_state)
                x: Node features [num_nodes, hidden_dim]
                edge_index: Edge connectivity [2, num_edges]
                edge_attr: Edge features [num_edges, edge_dim]
                hidden_state: GRU hidden state [1, num_nodes, hidden_dim]

        Returns:
            Tuple of (node_output, new_hidden_state)
        """
        x, edge_index, edge_attr, hidden_state = inputs

        # Get source and target node indices
        row, col = edge_index[0], edge_index[1]

        # Transform edge features to message weights
        edge_weights = self.edge_network(edge_attr)
        edge_weights = tf.reshape(edge_weights, [-1, self.hidden_dim, self.hidden_dim])

        # Compute messages
        neighbor_features = tf.gather(x, col)
        messages = tf.einsum('bij,bj->bi', edge_weights, neighbor_features)

        # Aggregate messages
        aggregated = tf.scatter_nd(
            tf.expand_dims(row, 1),
            messages,
            shape=[tf.shape(x)[0], self.hidden_dim]
        )

        # Update states with GRU
        aggregated = tf.expand_dims(aggregated, 0)  # Add batch dimension for GRU
        output, new_state = self.gru(aggregated, initial_state=hidden_state)
        output = tf.squeeze(output, 0)  # Remove batch dimension

        return output, new_state


class MPNN(Model):
    """Message Passing Neural Network for crystal property prediction.

    This model implements the MPNN architecture with GRU-based message passing,
    edge networks, and flexible pooling strategies.

    Key Features:
    - Edge network for computing interaction-specific message functions
    - GRU-based node state updates
    - Multiple message passing layers
    - Flexible graph-level pooling
    - Optional batch normalization and dropout
    """

    def __init__(self,
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 num_conv_layers: int = 3,
                 num_dense_layers: int = 2,
                 dropout_rate: float = 0.0,
                 pool_method: str = 'mean',
                 use_batch_norm: bool = True,
                 **kwargs):
        super(MPNN, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_conv_layers = num_conv_layers
        self.pool_method = pool_method

        # Initial node embedding
        self.node_embedding = Sequential([
            layers.Dense(hidden_dim),
            layers.ReLU()
        ])

        # Message passing layers
        self.conv_layers = []
        self.batch_norms = []

        for _ in range(num_conv_layers):
            self.conv_layers.append(
                MPNNConvolution(hidden_dim, edge_features)
            )
            if use_batch_norm:
                self.batch_norms.append(layers.BatchNormalization())

        # Output network
        dense_layers = []
        for _ in range(num_dense_layers):
            dense_layers.extend([
                layers.Dense(hidden_dim),
                layers.ReLU(),
                layers.Dropout(dropout_rate)
            ])
        dense_layers.append(layers.Dense(output_dim))
        self.output_network = Sequential(dense_layers)

    def pool_nodes(self, x: tf.Tensor, batch_idx: tf.Tensor) -> tf.Tensor:
        """Pool node features to graph-level representations.

        Args:
            x: Node features [num_nodes, channels]
            batch_idx: Batch assignments [num_nodes]

        Returns:
            Graph-level features [batch_size, channels]
        """
        num_graphs = tf.reduce_max(batch_idx) + 1

        if self.pool_method == 'mean':
            return tf.scatter_nd(
                tf.expand_dims(batch_idx, 1),
                x,
                shape=[num_graphs, self.hidden_dim]
            ) / tf.scatter_nd(
                tf.expand_dims(batch_idx, 1),
                tf.ones_like(batch_idx, dtype=tf.float32),
                shape=[num_graphs]
            )[:, None]

        elif self.pool_method == 'sum':
            return tf.scatter_nd(
                tf.expand_dims(batch_idx, 1),
                x,
                shape=[num_graphs, self.hidden_dim]
            )

        elif self.pool_method == 'max':
            return tf.scatter_nd(
                tf.expand_dims(batch_idx, 1),
                x,
                shape=[num_graphs, self.hidden_dim],
                reduction='max'
            )

        raise ValueError(f"Unknown pooling method: {self.pool_method}")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Tuple of (x, edge_index, edge_attr, batch_idx)
            training: Whether in training mode

        Returns:
            Predicted properties [batch_size, output_dim]
        """
        x, edge_index, edge_attr, batch_idx = inputs

        # Initial embedding
        x = self.node_embedding(x)

        # Initialize hidden state
        hidden_state = tf.zeros([1, tf.shape(x)[0], self.hidden_dim])

        # Message passing layers
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            # Message passing and state update
            x_new, hidden_state = conv((x, edge_index, edge_attr, hidden_state),
                                       training=training)

            # Batch norm and residual connection
            if bn is not None:
                x_new = bn(x_new, training=training)
            x = x + x_new

        # Pool nodes to graph representation
        x = self.pool_nodes(x, batch_idx)

        # Final prediction
        out = self.output_network(x, training=training)

        return out if self.output_dim > 1 else tf.squeeze(out, -1)

    def compute_loss(self, data, training: bool = False) -> tf.Tensor:
        """Compute model loss with gradient monitoring.

        Args:
            data: Tuple of (inputs, targets)
            training: Whether in training mode

        Returns:
            Loss value
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=training)
            loss = tf.reduce_mean(tf.square(y - y_pred))

        if training:
            # Compute and apply gradients with clipping
            gradients = tape.gradient(loss, self.trainable_variables)
            gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # Monitor gradient norm
            tf.summary.scalar('gradient_norm', grad_norm, step=self.optimizer.iterations)

        return loss

    @property
    def metrics(self):
        """Model metrics including MAE and RMSE."""
        return [
            keras.metrics.MeanAbsoluteError(name='mae'),
            keras.metrics.RootMeanSquaredError(name='rmse')
        ]