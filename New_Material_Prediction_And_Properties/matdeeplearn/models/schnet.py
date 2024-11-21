import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class ContinuousFilterConv(layers.Layer):
    """Continuous-filter convolution layer for SchNet.

    This layer implements interaction blocks with continuous filters,
    specifically designed for modeling quantum interactions in molecular systems.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_filters: int = 64,
                 cutoff: float = 8.0,
                 **kwargs):
        super(ContinuousFilterConv, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.cutoff = cutoff

        # Filter-generating network
        self.filter_net = Sequential([
            layers.Dense(num_filters, activation='tanh'),
            layers.Dense(num_filters),
        ])

        # Interaction network
        self.interaction_net = Sequential([
            layers.Dense(hidden_dim, activation='softplus'),
            layers.Dense(hidden_dim)
        ])

    def compute_filter(self, distances: tf.Tensor) -> tf.Tensor:
        """Compute continuous filters based on interatomic distances.

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

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Tuple of (x, edge_index, edge_weight, edge_attr)
                x: Node features [num_nodes, hidden_dim]
                edge_index: Edge connectivity [2, num_edges]
                edge_weight: Edge distances [num_edges]
                edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        x, edge_index, edge_weight, edge_attr = inputs

        # Get source and target node indices
        row, col = edge_index[0], edge_index[1]

        # Compute filter values
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

        # Apply interaction network
        return self.interaction_net(aggregated)


class SchNet(Model):
    """SchNet model for quantum chemistry and materials science.

    This model implements the SchNet architecture for predicting molecular and crystal
    properties using continuous-filter convolutions and quantum-chemical insights.

    Key Features:
    - Continuous-filter convolutions for modeling quantum interactions
    - Distance-based edge features with smooth cutoff
    - Multiple interaction blocks with residual connections
    - Physics-informed architecture design
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
                 use_batch_norm: bool = True,
                 **kwargs):
        super(SchNet, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cutoff = cutoff

        # Initial embedding
        self.embedding = layers.Dense(hidden_dim)

        # Interaction blocks
        self.interactions = []
        self.batch_norms = []

        for _ in range(num_interactions):
            self.interactions.append(
                ContinuousFilterConv(hidden_dim, num_filters, cutoff)
            )
            if use_batch_norm:
                self.batch_norms.append(layers.BatchNormalization())

        # Output network
        output_layers = []
        for _ in range(num_dense_layers):
            output_layers.extend([
                layers.Dense(hidden_dim // 2, activation='softplus'),
                layers.Dropout(dropout_rate)
            ])
        output_layers.append(layers.Dense(output_dim))
        self.output_network = Sequential(output_layers)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Tuple of (x, edge_index, edge_weight, edge_attr, batch_idx)
            training: Whether in training mode

        Returns:
            Predicted properties [batch_size, output_dim]
        """
        x, edge_index, edge_weight, edge_attr, batch_idx = inputs

        # Initial embedding
        x = self.embedding(x)

        # Interaction blocks with residual connections
        for interaction, bn in zip(self.interactions, self.batch_norms):
            x_interaction = interaction(
                (x, edge_index, edge_weight, edge_attr),
                training=training
            )
            if bn is not None:
                x_interaction = bn(x_interaction, training=training)
            x = x + x_interaction

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
        """Custom training step with gradient monitoring and clipping.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dict of metrics
        """
        inputs, targets = data

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compute_loss(predictions, targets)

        # Compute and apply gradients with clipping
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(targets, predictions)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({
            'loss': loss,
            'gradient_norm': grad_norm
        })

        return metrics

    def test_step(self, data):
        """Custom test step with uncertainty estimation.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dict of metrics
        """
        inputs, targets = data

        # Multiple forward passes for uncertainty estimation
        predictions = []
        for _ in range(10):  # Monte Carlo sampling with dropout
            pred = self(inputs, training=True)  # Enable dropout during inference
            predictions.append(pred)

        # Compute mean and variance
        pred_mean = tf.reduce_mean(predictions, axis=0)
        pred_var = tf.reduce_variance(predictions, axis=0)

        # Update metrics
        self.compiled_metrics.update_state(targets, pred_mean)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['prediction_uncertainty'] = tf.reduce_mean(pred_var)

        return metrics

    @property
    def metrics(self):
        """Model metrics including energy and force metrics."""
        return [
            keras.metrics.MeanAbsoluteError(name='energy_mae'),
            keras.metrics.RootMeanSquaredError(name='energy_rmse'),
            keras.metrics.MeanAbsoluteError(name='force_mae'),
            keras.metrics.RootMeanSquaredError(name='force_rmse')
        ]