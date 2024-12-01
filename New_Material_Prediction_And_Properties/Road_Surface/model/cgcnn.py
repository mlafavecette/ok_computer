import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, Tuple


class CGConvLayer(layers.Layer):
    def __init__(self, hidden_dim: int, activation: str = 'relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_update = layers.Dense(hidden_dim, activation=activation)
        self.edge_update = layers.Dense(hidden_dim, activation=activation)
        self.combine = layers.Dense(hidden_dim)
        self.norm = layers.BatchNormalization()

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False):
        x, edge_index, edge_attr = inputs
        row, col = edge_index[0], edge_index[1]

        # Edge update
        edge_features = self.edge_update(edge_attr)
        source_features = tf.gather(x, row)
        messages = source_features * edge_features

        # Aggregate messages
        output = tf.scatter_nd(
            tf.expand_dims(row, 1),
            messages,
            shape=[tf.shape(x)[0], self.hidden_dim]
        )

        # Node update
        output = self.node_update(output)
        output = self.norm(output, training=training)

        return output


class CGCNN(Model):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.node_embed = layers.Dense(config['hidden_dim'])

        # Convolutional layers
        self.conv_layers = [
            CGConvLayer(config['hidden_dim'])
            for _ in range(config['num_conv_layers'])
        ]

        # Output network
        self.output_layers = []
        for _ in range(config['num_dense_layers']):
            self.output_layers.extend([
                layers.Dense(config['hidden_dim'], activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(config['dropout_rate'])
            ])
        self.final_layer = layers.Dense(3)  # strength, porosity, permeability

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False):
        x, edge_index, edge_attr = inputs

        # Initial embedding
        x = self.node_embed(x)

        # Graph convolutions
        for conv in self.conv_layers:
            x_new = conv((x, edge_index, edge_attr), training=training)
            x = x + x_new  # Residual connection

        # Global pooling
        x = tf.reduce_mean(x, axis=0)

        # Output layers
        for layer in self.output_layers:
            x = layer(x, training=training)

        return self.final_layer(x)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}