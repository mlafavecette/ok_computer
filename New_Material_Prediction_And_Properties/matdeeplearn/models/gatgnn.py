import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class GlobalAttentionLayer(layers.Layer):
    """Global attention mechanism for crystal graphs.

    This layer computes attention weights by considering both local node features
    and global crystal features.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_fc_layers: int = 2,
                 dropout_rate: float = 0.0,
                 **kwargs):
        super(GlobalAttentionLayer, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_fc_layers = num_fc_layers
        self.dropout_rate = dropout_rate

        # Build MLP layers for attention computation
        self.attention_layers = []
        for i in range(num_fc_layers + 1):
            if i == 0:
                self.attention_layers.append(
                    layers.Dense(hidden_dim, name=f'att_fc_{i}')
                )
            elif i == num_fc_layers:
                self.attention_layers.append(
                    layers.Dense(1, name=f'att_fc_{i}')
                )
            else:
                self.attention_layers.append(
                    layers.Dense(hidden_dim, name=f'att_fc_{i}')
                )

        self.batch_norm_layers = [
            layers.BatchNormalization() for _ in range(num_fc_layers)
        ]

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        """Compute attention weights for each node.

        Args:
            inputs: Tuple of (node_features, batch_idx, global_features)
            training: Whether in training mode

        Returns:
            Attention weights for each node [num_nodes, 1]
        """
        x, batch_idx, global_features = inputs

        # Expand global features to match node features
        global_features = tf.gather(global_features, batch_idx)

        # Concatenate node and global features
        features = tf.concat([x, global_features], axis=-1)

        # Compute attention weights through MLP
        out = features
        for i, (layer, bn) in enumerate(zip(self.attention_layers[:-1],
                                            self.batch_norm_layers)):
            out = layer(out)
            out = bn(out, training=training)
            out = tf.nn.relu(out)
            if training and self.dropout_rate > 0:
                out = tf.nn.dropout(out, self.dropout_rate)

        # Final projection to scalar attention weights
        attention = self.attention_layers[-1](out)

        # Normalize attention weights per graph
        attention = tf.exp(attention)  # Softmax numerator
        normalizer = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            attention,
            shape=[tf.reduce_max(batch_idx) + 1, 1]
        )
        attention = attention / tf.gather(normalizer, batch_idx)

        return attention


class AGATConvolution(layers.Layer):
    """Atomistic Graph Attention Convolution layer.

    This layer implements edge-aware graph attention mechanisms for crystal graphs.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout_rate: float = 0.0,
                 **kwargs):
        super(AGATConvolution, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Trainable parameters
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

        self.bn = layers.BatchNormalization()

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        """Forward pass of attention convolution.

        Args:
            inputs: Tuple of (x, edge_index, edge_attr)
            training: Whether in training mode

        Returns:
            Updated node features
        """
        x, edge_index, edge_attr = inputs

        # Get source and target node indices
        row, col = edge_index[0], edge_index[1]

        # Gather source and target node features
        x_i = tf.gather(x, row)
        x_j = tf.gather(x, col)

        # Concatenate with edge features
        x_i = tf.concat([x_i, edge_attr], axis=-1)
        x_j = tf.concat([x_j, edge_attr], axis=-1)

        # Linear transformation and reshape to heads
        x_i = tf.matmul(x_i, self.W)
        x_j = tf.matmul(x_j, self.W)
        x_i = tf.reshape(x_i, [-1, self.num_heads, self.hidden_dim])
        x_j = tf.reshape(x_j, [-1, self.num_heads, self.hidden_dim])

        # Compute attention coefficients
        alpha = tf.concat([x_i, x_j], axis=-1) * self.att
        alpha = tf.reduce_sum(alpha, axis=-1)
        alpha = tf.nn.leaky_relu(alpha, alpha=0.2)
        alpha = self.bn(alpha, training=training)

        # Normalize attention coefficients
        alpha = tf.exp(alpha)  # Softmax numerator
        normalizer = tf.scatter_nd(
            tf.expand_dims(row, 1),
            alpha,
            shape=[tf.shape(x)[0], self.num_heads]
        )
        alpha = alpha / tf.gather(normalizer, row)

        if training and self.dropout_rate > 0:
            alpha = tf.nn.dropout(alpha, self.dropout_rate)

        # Apply attention weights and aggregate
        out = x_j * tf.expand_dims(alpha, -1)
        out = tf.transpose(out, [1, 0, 2])  # [num_heads, num_edges, hidden_dim]

        # Aggregate messages
        out = tf.scatter_nd(
            tf.expand_dims(row, 1),
            tf.reshape(out, [-1, self.num_heads * self.hidden_dim]),
            shape=[tf.shape(x)[0], self.num_heads * self.hidden_dim]
        )

        # Average over heads and add bias
        out = tf.reshape(out, [-1, self.num_heads, self.hidden_dim])
        out = tf.reduce_mean(out, axis=1)
        out = out + self.bias

        return out


class GATGNN(Model):
    """Graph Attention Neural Network for crystal property prediction.

    This model combines edge-conditioned attention mechanisms with global attention
    for effective learning on crystal structures.
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
                 **kwargs):
        super(GATGNN, self).__init__(**kwargs)

        # Model parameters
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.output_dim = output_dim

        # Initial embeddings
        self.node_embedding = Sequential([
            layers.Dense(hidden_dim),
            layers.LeakyReLU(0.2)
        ])

        self.edge_embedding = Sequential([
            layers.Dense(hidden_dim),
            layers.LeakyReLU(0.2)
        ])

        # Graph convolution layers
        self.conv_layers = [
            AGATConvolution(
                hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            ) for _ in range(num_conv_layers)
        ]

        # Global attention
        self.global_attention = GlobalAttentionLayer(
            hidden_dim,
            dropout_rate=dropout_rate
        )

        # Output layers
        self.output_layers = Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(output_dim)
        ])

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

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

        # Store initial representation for residual connections
        initial_x = x

        # Graph convolution layers with residual connections
        for conv in self.conv_layers:
            x_new = conv((x, edge_index, edge_attr), training=training)
            x = x + x_new

        # Apply global attention
        attention = self.global_attention(
            (x, batch_idx, global_features),
            training=training
        )
        x = x * attention

        # Pool nodes to graph representation
        num_graphs = tf.reduce_max(batch_idx) + 1
        out = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            x,
            shape=[num_graphs, self.hidden_dim]
        )

        # Final prediction
        out = self.output_layers(out, training=training)

        return out if self.output_dim > 1 else tf.squeeze(out, -1)

    def train_step(self, data):
        """Custom training step.

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