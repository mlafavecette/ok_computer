import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_conv_layers: int = 4
    num_dense_layers: int = 3
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32


class RoadMaterialsModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._build_model()

    def _build_model(self):
        # Input layers
        self.node_embed = layers.Dense(self.config.hidden_dim)
        self.edge_embed = layers.Dense(self.config.hidden_dim)

        # Graph conv layers
        self.conv_layers = []
        self.batch_norms = []
        for _ in range(self.config.num_conv_layers):
            self.conv_layers.append(
                layers.Dense(self.config.hidden_dim, activation='relu')
            )
            self.batch_norms.append(layers.BatchNormalization())

        # Output network
        self.output_layers = []
        for _ in range(self.config.num_dense_layers):
            self.output_layers.extend([
                layers.Dense(self.config.hidden_dim, activation='relu'),
                layers.Dropout(self.config.dropout_rate),
                layers.BatchNormalization()
            ])
        self.final_layer = layers.Dense(3)  # Strength, porosity, permeability

    def call(self, inputs: Tuple[tf.Tensor, ...], training=False):
        node_feat, edge_index, edge_feat = inputs

        # Initial embeddings
        x = self.node_embed(node_feat)
        edge_attr = self.edge_embed(edge_feat)

        # Graph convolutions
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            # Message passing
            row, col = edge_index[0], edge_index[1]
            messages = tf.gather(x, col) + edge_attr

            # Aggregate messages
            x_new = tf.scatter_nd(
                tf.expand_dims(row, 1),
                messages,
                shape=tf.shape(x)
            )

            # Update features
            x_new = conv(x_new)
            x_new = bn(x_new, training=training)
            x = x + x_new

        # Pool to graph level
        x = tf.reduce_mean(x, axis=0)

        # Output layers
        for layer in self.output_layers:
            x = layer(x, training=training)

        return self.final_layer(x)

    def train_step(self, data):
        inputs, targets = data

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(targets, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}