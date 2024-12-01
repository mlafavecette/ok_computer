import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, Tuple


class DiffGroupNorm(layers.Layer):
    def __init__(self, channels: int, num_groups: int = 10, epsilon: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.gamma = self.add_weight(shape=(channels,), initializer='ones')
        self.beta = self.add_weight(shape=(channels,), initializer='zeros')

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        shape = tf.shape(x)
        N = shape[0]
        C = shape[1]
        G = self.num_groups

        x = tf.reshape(x, [N, G, C // G])
        mean, var = tf.nn.moments(x, axes=[2], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, shape)

        return x * self.gamma + self.beta


class SuperCGConv(layers.Layer):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.edge_net = tf.keras.Sequential([
            layers.Dense(2 * hidden_dim, activation='relu'),
            layers.Dense(hidden_dim)
        ])
        self.node_net = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim)
        ])
        self.norm = DiffGroupNorm(hidden_dim)

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False):
        x, edge_index, edge_attr = inputs
        row, col = edge_index[0], edge_index[1]

        # Edge features
        edge_feat = self.edge_net(edge_attr)

        # Gather and combine
        neighbor_feat = tf.gather(x, col)
        messages = neighbor_feat * edge_feat

        # Aggregate
        out = tf.scatter_nd(
            tf.expand_dims(row, 1),
            messages,
            shape=tf.shape(x)
        )

        # Update and normalize
        out = self.node_net(out)
        out = self.norm(out, training=training)

        return out


class SuperCGCNN(Model):
    def __init__(self, config: Dict):
        super().__init__()
        hidden_dim = config['hidden_dim']

        self.node_embed = tf.keras.Sequential([
            layers.Dense(hidden_dim),
            layers.ReLU()
        ])

        self.conv_layers = []
        self.norms = []
        for _ in range(config['num_conv_layers']):
            self.conv_layers.append(
                SuperCGConv(hidden_dim, config['edge_features'])
            )
            self.norms.append(DiffGroupNorm(hidden_dim))

        self.output_net = []
        for _ in range(config['num_dense_layers']):
            self.output_net.extend([
                layers.Dense(hidden_dim, activation='relu'),
                layers.Dropout(config['dropout_rate'])
            ])
        self.final_layer = layers.Dense(3)

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False):
        x, edge_index, edge_attr = inputs

        # Initial embedding
        x = self.node_embed(x)
        initial_feat = x

        # Convolutions with residuals
        for conv, norm in zip(self.conv_layers, self.norms):
            x_conv = conv((x, edge_index, edge_attr), training=training)
            x_conv = norm(x_conv, training=training)
            x = x + x_conv

        # Global residual
        x = x + initial_feat

        # Pool and output
        x = tf.reduce_mean(x, axis=0)

        for layer in self.output_net:
            x = layer(x, training=training)

        return self.final_layer(x)