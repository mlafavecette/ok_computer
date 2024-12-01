import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, Tuple


class MEGNetBlock(layers.Layer):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='softplus'),
            layers.Dense(hidden_dim)
        ])
        self.node_mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='softplus'),
            layers.Dense(hidden_dim)
        ])
        self.global_mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='softplus'),
            layers.Dense(hidden_dim)
        ])
        self.combine = layers.Dense(hidden_dim)

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False):
        x, edge_index, edge_attr, global_state = inputs
        row, col = edge_index[0], edge_index[1]

        # Edge update
        src_features = tf.gather(x, row)
        dst_features = tf.gather(x, col)
        edge_inputs = tf.concat([edge_attr, src_features, dst_features], axis=-1)
        edge_attr_updated = self.edge_mlp(edge_inputs)

        # Node update
        node_messages = tf.scatter_nd(
            tf.expand_dims(row, 1),
            edge_attr_updated,
            shape=[tf.shape(x)[0], edge_attr_updated.shape[-1]]
        )
        node_inputs = tf.concat([x, node_messages], axis=-1)
        x_updated = self.node_mlp(node_inputs)

        # Global update
        global_inputs = tf.concat([
            tf.reduce_mean(x_updated, axis=0, keepdims=True),
            tf.reduce_mean(edge_attr_updated, axis=0, keepdims=True),
            global_state
        ], axis=-1)
        global_updated = self.global_mlp(global_inputs)

        return x_updated, edge_attr_updated, global_updated


class MEGNet(Model):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Initial embeddings
        self.node_embed = layers.Dense(config['hidden_dim'])
        self.edge_embed = layers.Dense(config['hidden_dim'])
        self.global_embed = layers.Dense(config['hidden_dim'])

        # MEGNet blocks
        self.blocks = [
            MEGNetBlock(config['hidden_dim'])
            for _ in range(config['num_blocks'])
        ]

        # Output network
        self.output_layers = []
        for _ in range(config['num_dense_layers']):
            self.output_layers.extend([
                layers.Dense(config['hidden_dim'], activation='softplus'),
                layers.BatchNormalization(),
                layers.Dropout(config['dropout_rate'])
            ])
        self.final_layer = layers.Dense(3)

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False):
        x, edge_index, edge_attr, global_state = inputs

        # Initial embeddings
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        global_state = self.global_embed(global_state)

        # Process through MEGNet blocks
        for block in self.blocks:
            x_new, edge_attr_new, global_new = block(
                (x, edge_index, edge_attr, global_state),
                training=training
            )
            # Residual connections
            x = x + x_new
            edge_attr = edge_attr + edge_attr_new
            global_state = global_state + global_new

        # Final prediction
        x = tf.concat([
            tf.reduce_mean(x, axis=0),
            global_state[0]
        ], axis=-1)

        for layer in self.output_layers:
            x = layer(x, training=training)

        return self.final_layer(x)