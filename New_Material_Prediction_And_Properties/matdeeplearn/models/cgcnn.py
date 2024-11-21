import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class CGConvLayer(layers.Layer):
    """Crystal Graph Convolutional Layer implementation in TensorFlow.

    This layer implements the core convolution operation for crystal graphs,
    processing both node features and edge features to update node representations.
    """

    def __init__(self,
                 out_channels: int,
                 edge_channels: int,
                 aggregation: str = 'mean',
                 use_bias: bool = True,
                 **kwargs):
        super(CGConvLayer, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.aggregation = aggregation
        self.use_bias = use_bias

        # Initialize trainable parameters
        self.weight = self.add_weight(
            shape=(out_channels + edge_channels, out_channels),
            initializer='glorot_uniform',
            trainable=True,
            name='weight'
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(out_channels,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )

    def build(self, input_shape):
        super(CGConvLayer, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Tuple of (x, edge_index, edge_attr)
                x: Node feature matrix [num_nodes, in_channels]
                edge_index: Edge connectivity [2, num_edges]
                edge_attr: Edge feature matrix [num_edges, edge_channels]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        x, edge_index, edge_attr = inputs

        # Get source and target node indices
        row, col = edge_index[0], edge_index[1]

        # Gather source and target node features
        x_i = tf.gather(x, row)
        x_j = tf.gather(x, col)

        # Combine node and edge features
        out = tf.concat([x_j, edge_attr], axis=-1)
        out = tf.matmul(out, self.weight)

        # Apply aggregation
        if self.aggregation == 'mean':
            out = tf.scatter_nd(
                tf.expand_dims(row, 1),
                out,
                shape=[tf.shape(x)[0], self.out_channels]
            )
            # Normalize by counting neighbors
            counts = tf.scatter_nd(
                tf.expand_dims(row, 1),
                tf.ones_like(row, dtype=tf.float32),
                shape=[tf.shape(x)[0]]
            )
            counts = tf.clip_by_value(counts, clip_value_min=1.0, clip_value_max=float('inf'))
            out = out / tf.expand_dims(counts, -1)

        if self.use_bias:
            out = out + self.bias

        return out


class CGCNN(Model):
    """Crystal Graph Convolutional Neural Network implemented in TensorFlow.

    This model implements the CGCNN architecture for learning on crystal structures,
    featuring multiple graph convolution layers, batch normalization, and flexible pooling.

    Args:
        node_features: Number of input node features
        edge_features: Number of input edge features
        hidden_channels: Number of hidden channels in conv layers
        num_conv_layers: Number of crystal graph conv layers
        num_hidden_layers: Number of hidden layers after pooling
        output_dim: Dimension of output (e.g. 1 for regression)
        pool_method: Method for graph pooling ('mean', 'sum', 'max')
        dropout_rate: Dropout rate for regularization
        batch_norm: Whether to use batch normalization
    """

    def __init__(self,
                 node_features: int,
                 edge_features: int,
                 hidden_channels: int = 64,
                 num_conv_layers: int = 3,
                 num_hidden_layers: int = 1,
                 output_dim: int = 1,
                 pool_method: str = 'mean',
                 dropout_rate: float = 0.0,
                 batch_norm: bool = True,
                 **kwargs):
        super(CGCNN, self).__init__(**kwargs)

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.pool_method = pool_method
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        # Initial embedding layer
        self.embedding = layers.Dense(hidden_channels, activation='relu')

        # Graph convolution layers
        self.conv_layers = []
        self.batch_norm_layers = []

        for _ in range(num_conv_layers):
            self.conv_layers.append(
                CGConvLayer(hidden_channels, edge_features)
            )
            if batch_norm:
                self.batch_norm_layers.append(layers.BatchNormalization())

        # Post-processing layers
        self.post_layers = []
        current_dim = hidden_channels

        for _ in range(num_hidden_layers):
            self.post_layers.extend([
                layers.Dense(current_dim, activation='relu'),
                layers.Dropout(dropout_rate)
            ])

        # Output layer
        self.output_layer = layers.Dense(output_dim)

    def pool_nodes(self, x: tf.Tensor, batch_idx: tf.Tensor) -> tf.Tensor:
        """Pools node features according to batch assignment.

        Args:
            x: Node features [num_nodes, channels]
            batch_idx: Batch assignment for each node [num_nodes]

        Returns:
            Pooled features [batch_size, channels]
        """
        num_graphs = tf.reduce_max(batch_idx) + 1

        if self.pool_method == 'mean':
            # Compute mean of node features per graph
            return tf.scatter_nd(
                tf.expand_dims(batch_idx, 1),
                x,
                shape=[num_graphs, self.hidden_channels]
            ) / tf.scatter_nd(
                tf.expand_dims(batch_idx, 1),
                tf.ones_like(batch_idx, dtype=tf.float32),
                shape=[num_graphs]
            )[:, None]

        elif self.pool_method == 'sum':
            return tf.scatter_nd(
                tf.expand_dims(batch_idx, 1),
                x,
                shape=[num_graphs, self.hidden_channels]
            )

        elif self.pool_method == 'max':
            return tf.scatter_nd(
                tf.expand_dims(batch_idx, 1),
                x,
                shape=[num_graphs, self.hidden_channels],
                reduction='max'
            )

        raise ValueError(f"Unknown pooling method: {self.pool_method}")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Tuple of (x, edge_index, edge_attr, batch_idx)
                x: Node features [num_nodes, node_features]
                edge_index: Edge connectivity [2, num_edges]
                edge_attr: Edge features [num_edges, edge_features]
                batch_idx: Batch assignments [num_nodes]
            training: Whether in training mode

        Returns:
            Predicted properties [batch_size, output_dim]
        """
        x, edge_index, edge_attr, batch_idx = inputs

        # Initial embedding
        x = self.embedding(x)

        # Graph convolution layers
        for conv, bn in zip(self.conv_layers, self.batch_norm_layers):
            x_new = conv((x, edge_index, edge_attr))
            if self.batch_norm:
                x_new = bn(x_new, training=training)
            x_new = tf.nn.relu(x_new)
            if self.dropout_rate > 0 and training:
                x_new = tf.nn.dropout(x_new, self.dropout_rate)
            x = x + x_new  # Residual connection

        # Pool nodes to graph representation
        x = self.pool_nodes(x, batch_idx)

        # Post-processing
        for layer in self.post_layers:
            x = layer(x, training=training)

        # Output projection
        out = self.output_layer(x)

        return out if self.output_dim > 1 else tf.squeeze(out, -1)

    def train_step(self, data):
        """Custom training step with gradient clipping.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dict of metrics
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        # Compute and apply gradients with clipping
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss

        return metrics