import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, Tuple


class ContinuousFilterConv(layers.Layer):
    def __init__(self, hidden_dim: int, num_filters: int, cutoff: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.cutoff = cutoff

        self.filter_net = tf.keras.Sequential([
            layers.Dense(num_filters, activation='tanh'),
            layers.Dense(num_filters)
        ])

        self.interaction_net = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='softplus'),
            layers.Dense(hidden_dim)
        ])

    def compute_filter(self, distances: tf.Tensor) -> tf.Tensor:
        scaled_dist = distances * (2.0 / self.cutoff) - 1.0
        filters = self.filter_net(tf.expand_dims(scaled_dist, -1))

        cutoff_vals = 0.5 * (tf.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoff_vals = tf.where(distances <= self.cutoff, cutoff_vals, 0.0)

        return filters * tf.expand_dims(cutoff_vals, -1)

    def call(self, inputs: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        x, edge_index, distances, edge_attr = inputs
        row, col = edge_index[0], edge_index[1]

        # Compute filters
        filters = self.compute_filter(distances)

        # Apply interaction
        neighbor_features = tf.gather(x, col)
        filtered_features = tf.expand_dims(neighbor_features, -2) * \
                            tf.expand_dims(filters, -1)

        # Aggregate messages
        messages = tf.reduce_sum(filtered_features, axis=1)
        output = tf.scatter_nd(
            tf.expand_dims(row, 1),
            messages,
            shape=[tf.shape(x)[0], self.hidden_dim]
        )

        return self.interaction_net(output)


class SchNet(Model):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Initial embedding
        self.embedding = layers.Dense(config['hidden_dim'])

        # Interaction blocks
        self.interactions = []
        self.batch_norms = []
        for _ in range(config['num_interactions']):
            self.interactions.append(
                ContinuousFilterConv(
                    config['hidden_dim'],
                    config['num_filters'],
                    config['cutoff']
                )
            )
            self.batch_norms.append(layers.BatchNormalization())

        # Output network
        self.output_layers = []
        for _ in range(config['num_dense_layers']):
            self.output_layers.extend([
                layers.Dense(config['hidden_dim'] // 2, activation='softplus'),
                layers.BatchNormalization(),
                layers.Dropout(config['dropout_rate'])
            ])
        self.final_layer = layers.Dense(3)

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False):
        x, edge_index, distances, edge_attr = inputs

        # Initial embedding
        x = self.embedding(x)

        # Interaction blocks
        for interaction, bn in zip(self.interactions, self.batch_norms):
            x_new = interaction((x, edge_index, distances, edge_attr))
            x_new = bn(x_new, training=training)
            x = x + x_new

        # Global pooling
        x = tf.reduce_mean(x, axis=0)

        # Output network
        for layer in self.output_layers:
            x = layer(x, training=training)

        return self.final_layer(x)