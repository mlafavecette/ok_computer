import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class SuperInteractionBlock(layers.Layer):
    """Enhanced interaction block for SchNet.

    This layer implements an improved version of SchNet interactions
    with better feature processing and stability.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_filters: int = 64,
                 cutoff: float = 8.0,
                 use_group_norm: bool = True,
                 **kwargs):
        super(SuperInteractionBlock, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.cutoff = cutoff

        # Filter-generating network
        self.filter_net = Sequential([
            layers.Dense(num_filters, activation='tanh'),
            layers.Dense(num_filters)
        ])

        # Interaction network
        self.interaction_net = Sequential([
            layers.Dense(hidden_dim, activation='softplus'),
            layers.Dense(hidden_dim)
        ])

        # Normalization
        if use_group_norm:
            self.norm = DiffGroupNorm(hidden_dim, 10)
        else:
            self.norm = layers.BatchNormalization()

    def compute_filter(self, distances: tf.Tensor) -> tf.Tensor:
        """Compute continuous filters with smooth cutoff.

        Args:
            distances: Interatomic distances [num_edges]

        Returns:
            Filter values [num_edges, num_filters]
        """
        # Scale distances
        scaled_dist = distances * (2.0 / self.cutoff) - 1.0

        # Compute filter values
        filters = self.filter_net(tf.expand_dims(scaled_dist, -1))

        # Apply smooth cutoff
        cutoff_vals = 0.5 * (tf.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoff_vals = tf.where(distances <= self.cutoff, cutoff_vals, 0.0)

        return filters * tf.expand_dims(cutoff_vals, -1)

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced interaction processing.

        Args:
            inputs: Tuple of (x, edge_index, edge_weight, edge_attr)
            training: Whether in training mode

        Returns:
            Updated node features
        """
        x, edge_index, edge_weight, edge_attr = inputs

        # Get source and target node indices
        row, col = edge_index[0], edge_index[1]

        # Compute interaction filters
        filters = self.compute_filter(edge_weight)

        # Get neighbor features and apply filters
        neighbor_features = tf.gather(x, col)
        filtered_features = tf.expand_dims(neighbor_features, -2) * \
                            tf.expand_dims(filters, -1)

        # Sum over filter dimension
        messages = tf.reduce_sum(filtered_features, axis=1)

        # Aggregate messages
        aggregated = tf.scatter_nd(
            tf.expand_dims(row, 1),
            messages,
            shape=[tf.shape(x)[0], self.hidden_dim]
        )

        # Apply interaction network and normalize
        out = self.interaction_net(aggregated)
        out = self.norm(out, training=training)

        return out


class SuperSchNet(Model):
    """Enhanced SchNet model for quantum chemistry and materials science.

    This model implements an improved version of SchNet with:
    - Enhanced continuous-filter convolutions
    - Advanced residual connections
    - Improved gradient flow
    - Better normalization scheme
    - Physics-informed architectural choices
    """

    def __init__(self,
                 node_features: int,
                 hidden_dim: int = 64,
                 num_filters: int = 64,
                 num_interactions: int = 3,
                 num_dense_layers: int = 2,
                 cutoff: float = 8.0,
                 output_dim: int = 1,
                 dropout_rate: float = 0.0,
                 use_group_norm: bool = True,
                 **kwargs):
        super(SuperSchNet, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cutoff = cutoff

        # Initial embeddings with normalization
        self.embedding = Sequential([
            layers.Dense(hidden_dim),
            DiffGroupNorm(hidden_dim, 10) if use_group_norm else layers.BatchNormalization(),
            layers.ReLU()
        ])

        # Interaction blocks
        self.interactions = []
        for _ in range(num_interactions):
            self.interactions.append(
                SuperInteractionBlock(
                    hidden_dim,
                    num_filters,
                    cutoff,
                    use_group_norm
                )
            )

        # Output network
        output_layers = []
        for _ in range(num_dense_layers):
            output_layers.extend([
                layers.Dense(hidden_dim // 2, activation='softplus'),
                layers.Dropout(dropout_rate)
            ])
        output_layers.append(layers.Dense(output_dim))
        self.output_network = Sequential(output_layers)

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Forward pass with enhanced feature processing.

        Args:
            inputs: Tuple of (node_attr, edge_index, edge_weight, edge_attr, batch_idx)
            training: Whether in training mode

        Returns:
            Predicted properties
        """
        x, edge_index, edge_weight, edge_attr, batch_idx = inputs

        # Initial embedding
        x = self.embedding(x)

        # Store initial features for global residual
        initial_features = x

        # Process through interaction blocks with residual connections
        for interaction in self.interactions:
            x_interaction = interaction(
                (x, edge_index, edge_weight, edge_attr),
                training=training
            )

            # Local residual connection
            x = x + x_interaction

        # Global residual connection
        x = x + initial_features

        # Pool atoms to molecule/crystal
        num_graphs = tf.reduce_max(batch_idx) + 1
        pooled = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            x,
            shape=[num_graphs, self.hidden_dim]
        )

        # Final prediction
        out = self.output_network(pooled, training=training)
        return out if self.output_dim > 1 else tf.squeeze(out, -1)

    def compute_energy_and_forces(self, inputs: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute energy and forces using gradient of energy w.r.t positions.

        Args:
            inputs: Model inputs including atomic positions

        Returns:
            Tuple of (energy, forces)
        """
        x, positions, edge_index, edge_attr, batch_idx = inputs

        with tf.GradientTape() as tape:
            tape.watch(positions)

            # Compute distances from positions
            row, col = edge_index[0], edge_index[1]
            distances = tf.norm(
                tf.gather(positions, row) - tf.gather(positions, col),
                axis=-1
            )

            # Predict energy
            energy = self((x, edge_index, distances, edge_attr, batch_idx))

        # Forces are negative gradient of energy w.r.t positions
        forces = -tape.gradient(energy, positions)

        return energy, forces

    def train_step(self, data):
        """Custom training step with physics-informed regularization.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dict of metrics
        """
        inputs, targets = data

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)

            # Main prediction loss
            main_loss = self.compiled_loss(targets, predictions)

            # Physics-informed regularization
            if hasattr(self, 'physics_regularization'):
                reg_loss = self.physics_regularization(inputs, predictions)
                total_loss = main_loss + self.reg_weight * reg_loss
            else:
                total_loss = main_loss
                reg_loss = 0.0

        # Compute and clip gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(targets, predictions)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({
            'loss': total_loss,
            'main_loss': main_loss,
            'reg_loss': reg_loss,
            'gradient_norm': grad_norm
        })

        return metrics

    @property
    def metrics(self):
        """Model metrics with physics-informed components."""
        return [
            keras.metrics.MeanAbsoluteError(name='energy_mae'),
            keras.metrics.RootMeanSquaredError(name='energy_rmse'),
            keras.metrics.MeanAbsoluteError(name='force_mae'),
            keras.metrics.RootMeanSquaredError(name='force_rmse')
        ]
