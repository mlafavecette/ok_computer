import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class MaterialGraphLayer(tf.keras.layers.Layer):
    """Graph convolution layer for materials property prediction."""

    def __init__(
            self,
            units: int,
            activation: str = 'silu',
            dropout: float = 0.1
    ):
        super().__init__()
        self.units = units
        self.dense = tf.keras.layers.Dense(units)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.activation = tf.keras.activations.get(activation)

    def call(
            self,
            node_features: tf.Tensor,
            edge_index: tf.Tensor,
            edge_features: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        # Message passing
        source, target = tf.unstack(edge_index, axis=0)
        messages = tf.gather(node_features, source)
        messages = self.dense(tf.concat([messages, edge_features], axis=-1))
        messages = self.activation(messages)

        # Aggregation
        output = tf.zeros_like(node_features)
        output = tf.tensor_scatter_nd_add(
            output,
            tf.expand_dims(target, 1),
            messages
        )

        output = self.norm(output)
        return self.dropout(output, training=training)


class MaterialGNN(tf.keras.Model):
    """Graph neural network for materials property prediction."""

    def __init__(
            self,
            hidden_dim: int = 256,
            num_layers: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()
        self.node_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('silu'),
            tf.keras.layers.Dropout(dropout)
        ])

        self.edge_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('silu'),
            tf.keras.layers.Dropout(dropout)
        ])

        self.graph_layers = [
            MaterialGraphLayer(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ]

        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        x = self.node_encoder(inputs['node_features'], training=training)
        edge_attr = self.edge_encoder(inputs['edge_attr'], training=training)

        for layer in self.graph_layers:
            x = x + layer(x, inputs['edge_index'], edge_attr, training=training)

        return self.global_pool(x)


class VAE(tf.keras.Model):
    """Variational autoencoder for material properties."""

    def __init__(
            self,
            latent_dim: int = 64,
            hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        for dim in hidden_dims:
            encoder_layers.extend([
                tf.keras.layers.Dense(dim),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation('silu')
            ])
        self.encoder = tf.keras.Sequential(encoder_layers)
        self.mean = tf.keras.layers.Dense(latent_dim)
        self.logvar = tf.keras.layers.Dense(latent_dim)

        # Decoder
        decoder_layers = []
        for dim in reversed(hidden_dims):
            decoder_layers.extend([
                tf.keras.layers.Dense(dim),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation('silu')
            ])
        self.decoder = tf.keras.Sequential(decoder_layers)

    def encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.encoder(x)
        return self.mean(x), self.logvar(x)

    def reparameterize(self, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=mean.shape)
        return mean + tf.exp(0.5 * logvar) * eps

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        return self.decoder(z)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar


class CycleGAN(tf.keras.Model):
    """CycleGAN for material property translation."""

    def __init__(self, input_shape: Tuple[int, ...]):
        super().__init__()
        self.gen_A2B = self._build_generator(input_shape)
        self.gen_B2A = self._build_generator(input_shape)
        self.disc_A = self._build_discriminator(input_shape)
        self.disc_B = self._build_discriminator(input_shape)

    def _build_generator(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        return tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(input_shape[-1], activation='tanh')
        ])

    def _build_discriminator(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        return tf.keras.Sequential([
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        real_A, real_B = inputs

        # Forward cycle
        fake_B = self.gen_A2B(real_A)
        cycle_A = self.gen_B2A(fake_B)

        # Backward cycle
        fake_A = self.gen_B2A(real_B)
        cycle_B = self.gen_A2B(fake_A)

        return {
            'fake_B': fake_B,
            'cycle_A': cycle_A,
            'fake_A': fake_A,
            'cycle_B': cycle_B
        }


def create_model(config: Dict) -> tf.keras.Model:
    """Model factory function."""
    model_type = config.get('type', 'gnn')

    if model_type == 'gnn':
        return MaterialGNN(**config.get('model_params', {}))
    elif model_type == 'vae':
        return VAE(**config.get('model_params', {}))
    elif model_type == 'cyclegan':
        return CycleGAN(config['input_shape'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")