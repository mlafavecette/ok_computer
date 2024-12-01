import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, Tuple


class FilterGenerator(layers.Layer):
    def __init__(self, num_filters: int, cutoff: float):
        super().__init__()
        self.num_filters = num_filters
        self.cutoff = cutoff

        self.filter_net = tf.keras.Sequential([
            layers.Dense(num_filters, activation='tanh'),
            layers.Dense(num_filters),
            DiffGroupNorm(num_filters)
        ])

    def call(self, distances: tf.Tensor, training: bool = False):
        scaled_dist = distances * (2.0 / self.cutoff) - 1.0
        filters = self.filter_net(tf.expand_dims(scaled_dist, -1), training=training)

        # Smooth cutoff
        cutoff_vals = 0.5 * (tf.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoff_vals = tf.where(distances <= self.cutoff, cutoff_vals, 0.0)

        return filters * tf.expand_dims(cutoff_vals, -1)


class SuperSchNetBlock(layers.Layer):
    def __init__(self, hidden_dim: int, num_filters: int, cutoff: float):
        super().__init__()
        self.filter_gen = FilterGenerator(num_filters, cutoff)
        self.interaction = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='softplus'),
            layers.Dense(hidden_dim),
            DiffGroupNorm(hidden_dim)
        ])

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False):
        x, edge_index, distances, edge_attr = inputs
        row, col = edge_index[0], edge_index[1]

        # Generate filters
        filters = self.filter_gen(distances, training=training)

        # Interaction
        neighbor_feat = tf.gather(x, col)
        filtered_feat = tf.expand_dims(neighbor_feat, -2) * tf.expand_dims(filters, -1)
        messages = tf.reduce_sum(filtered_feat, axis=1)

        # Aggregate
        out = tf.scatter_nd(
            tf.expand_dims(row, 1),
            messages,
            shape=tf.shape(x)
        )

        return self.interaction(out, training=training)


class SuperSchNet(Model):
    def __init__(self, config: Dict):
        super().__init__()
        self.embedding = layers.Dense(config['hidden_dim'])

        # Interaction blocks
        self.blocks = []
        for _ in range(config['num_blocks']):
            self.blocks.append(
                SuperSchNetBlock(
                    config['hidden_dim'],
                    config['num_filters'],
                    config['cutoff']
                )
            )

        # Output network
        self.output_net = []
        for _ in range(config['num_dense_layers']):
            self.output_net.extend([
                layers.Dense(config['hidden_dim'], activation='softplus'),
                DiffGroupNorm(config['hidden_dim']),
                layers.Dropout(config['dropout_rate'])
            ])
        self.final_layer = layers.Dense(3)

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False):
        x, edge_index, distances, edge_attr = inputs

        # Initial embedding
        x = self.embedding(x)

        # Interaction blocks with residuals
        for block in self.blocks:
            x_new = block((x, edge_index, distances, edge_attr), training=training)
            x = x + x_new

        # Pool and predict
        x = tf.reduce_mean(x, axis=0)

        for layer in self.output_net:
            x = layer(x, training=training)

        return self.final_layer(x)