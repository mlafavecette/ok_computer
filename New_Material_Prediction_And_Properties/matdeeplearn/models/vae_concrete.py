import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import List, Tuple, Dict


class ConcreteEncoder(layers.Layer):
    """Encoder network for concrete composition analysis."""

    def __init__(
            self,
            latent_dim: int = 32,
            hidden_dims: List[int] = [256, 128, 64],
            name: str = "encoder",
            **kwargs
    ):
        super(ConcreteEncoder, self).__init__(name=name, **kwargs)

        self.latent_dim = latent_dim
        self.hidden_layers = []

        for dim in hidden_dims:
            self.hidden_layers.extend([
                layers.Dense(
                    dim,
                    activation='swish',
                    kernel_initializer='glorot_normal'
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.1)
            ])

        self.mu = layers.Dense(latent_dim, name='mu')
        self.log_var = layers.Dense(latent_dim, name='log_var')

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.mu(x), self.log_var(x)


class ConcreteDecoder(layers.Layer):
    """Decoder network for concrete composition generation."""

    def __init__(
            self,
            original_dim: int,
            hidden_dims: List[int] = [64, 128, 256],
            name: str = "decoder",
            **kwargs
    ):
        super(ConcreteDecoder, self).__init__(name=name, **kwargs)

        self.hidden_layers = []

        for dim in hidden_dims:
            self.hidden_layers.extend([
                layers.Dense(
                    dim,
                    activation='swish',
                    kernel_initializer='glorot_normal'
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.1)
            ])

        self.output_layer = layers.Dense(
            original_dim,
            activation='sigmoid',
            name='output'
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x)


class ConcreteVAE(Model):
    """VAE for concrete composition optimization."""

    def __init__(
            self,
            original_dim: int,
            latent_dim: int = 32,
            encoder_hidden_dims: List[int] = [256, 128, 64],
            decoder_hidden_dims: List[int] = [64, 128, 256],
            beta: float = 1.0,
            name: str = "concrete_vae",
            **kwargs
    ):
        super(ConcreteVAE, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = ConcreteEncoder(latent_dim, encoder_hidden_dims)
        self.decoder = ConcreteDecoder(original_dim, decoder_hidden_dims)

        self.strength_predictor = layers.Dense(3, name='strength_prediction')
        self.carbon_predictor = layers.Dense(2, name='carbon_prediction')

        # Initialize metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.strength_loss_tracker = tf.keras.metrics.Mean(name="strength_loss")
        self.carbon_loss_tracker = tf.keras.metrics.Mean(name="carbon_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.strength_loss_tracker,
            self.carbon_loss_tracker
        ]

    def _reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE training."""
        std = tf.exp(log_var * 0.5)
        eps = tf.random.normal(shape=mu.shape)
        return mu + eps * std

    def call(self, inputs, training=False):
        """Forward pass through the VAE."""
        mu, log_var = self.encoder(inputs, training=training)
        z = self._reparameterize(mu, log_var)
        reconstruction = self.decoder(z, training=training)
        strength_pred = self.strength_predictor(z)
        carbon_pred = self.carbon_predictor(z)
        return reconstruction, strength_pred, carbon_pred, mu, log_var

    def train_step(self, data):
        """Custom training step with multiple loss components."""
        composition, targets = data

        with tf.GradientTape() as tape:
            reconstruction, strength_pred, carbon_pred, mu, log_var = self(composition, training=True)

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(composition, reconstruction),
                    axis=1
                )
            )

            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mu) - tf.exp(log_var)
            )

            # Property prediction losses
            strength_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(
                    targets[:, :3],
                    strength_pred
                )
            )

            carbon_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(
                    targets[:, 3:],
                    carbon_pred
                )
            )

            # Combined loss
            total_loss = (
                    reconstruction_loss +
                    self.beta * kl_loss +
                    2.0 * strength_loss +
                    1.5 * carbon_loss
            )

        # Compute and apply gradients
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.strength_loss_tracker.update_state(strength_loss)
        self.carbon_loss_tracker.update_state(carbon_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "strength_loss": self.strength_loss_tracker.result(),
            "carbon_loss": self.carbon_loss_tracker.result(),
        }

    def generate_optimized_mixture(
            self,
            target_strength: Tuple[float, float, float],
            target_carbon: Tuple[float, float],
            num_samples: int = 1000
    ) -> np.ndarray:
        """Generate concrete mixtures optimized for target properties."""
        z_samples = tf.random.normal((num_samples, self.latent_dim))

        compositions = self.decoder(z_samples)
        strengths = self.strength_predictor(z_samples)
        carbons = self.carbon_predictor(z_samples)

        strength_scores = tf.reduce_mean(
            tf.square(strengths - target_strength), axis=1
        )
        carbon_scores = tf.reduce_mean(
            tf.square(carbons - target_carbon), axis=1
        )

        total_scores = strength_scores + carbon_scores
        best_idx = tf.argmin(total_scores)

        return compositions[best_idx].numpy()