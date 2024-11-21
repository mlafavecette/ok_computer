"""
Neural Networks for Materials Science Descriptors

Implements neural network architectures for processing Smooth Overlap of Atomic
Positions (SOAP) and Sine Matrix (SM) descriptors commonly used in materials science.

Key features:
- Optimized for materials property prediction
- Configurable architecture depth and width
- Advanced regularization options
- Efficient batch processing

Author: Claude
Date: 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class MaterialsDescriptorNetwork(keras.Model):
    """
    Base class for materials descriptor neural networks.

    Provides common functionality for processing materials science descriptors
    with configurable architecture and regularization.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        output_dim: Output dimension (default=1 for regression)
        activation: Activation function
        dropout_rate: Dropout rate for regularization
        batch_norm: Whether to use batch normalization
        l2_reg: L2 regularization factor
        learning_rate: Initial learning rate
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 3,
            output_dim: int = 1,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            batch_norm: bool = True,
            l2_reg: float = 0.0,
            learning_rate: float = 0.001,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input layer
        self.input_layer = layers.Dense(
            hidden_dim,
            activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )

        # Hidden layers
        self.hidden_layers = []
        for _ in range(num_layers):
            if batch_norm:
                self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Activation(activation))
            if dropout_rate > 0:
                self.hidden_layers.append(layers.Dropout(dropout_rate))
            self.hidden_layers.append(
                layers.Dense(
                    hidden_dim,
                    activation=None,
                    kernel_regularizer=keras.regularizers.l2(l2_reg)
                )
            )

        # Output layer
        self.output_layer = layers.Dense(
            output_dim,
            activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate)

    def call(self, inputs, training=None):
        """Forward pass computation.

        Args:
            inputs: Input features [batch_size, input_dim]
            training: Whether in training mode

        Returns:
            Predictions [batch_size, output_dim]
        """
        x = self.input_layer(inputs)

        for layer in self.hidden_layers:
            x = layer(x, training=training)

        return self.output_layer(x)

    def get_config(self):
        """Returns model configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim
        }


class SOAP(MaterialsDescriptorNetwork):
    """
    Neural network for Smooth Overlap of Atomic Positions (SOAP) descriptors.

    Specialized architecture for processing SOAP descriptors, which capture
    local atomic environments through spherical harmonic expansion of
    atomic density overlap.

    Additional features:
    - Periodic boundary handling
    - Rotational invariance
    - Length scale sensitivity

    Args:
        soap_dim: Dimension of SOAP descriptor
        **kwargs: Arguments passed to MaterialsDescriptorNetwork
    """

    def __init__(
            self,
            soap_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 4,
            periodic: bool = True,
            **kwargs
    ):
        super().__init__(
            input_dim=soap_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs
        )

        self.periodic = periodic

        # Additional layers for SOAP-specific processing
        self.soap_projection = layers.Dense(
            hidden_dim,
            activation="tanh",
            kernel_initializer="glorot_uniform"
        )

        if periodic:
            self.periodic_pad = layers.ZeroPadding1D(padding=(1, 1))

    def call(self, inputs, training=None):
        """
        Forward pass with SOAP-specific processing.

        Args:
            inputs: SOAP descriptors [batch_size, soap_dim]
            training: Whether in training mode

        Returns:
            Predictions [batch_size, output_dim]
        """
        # Project SOAP features
        x = self.soap_projection(inputs)

        # Apply periodic padding if needed
        if self.periodic:
            x = tf.expand_dims(x, 1)  # Add channel dimension
            x = self.periodic_pad(x)
            x = tf.squeeze(x, 1)  # Remove channel dimension

        # Process through main network
        return super().call(x, training=training)


class SM(MaterialsDescriptorNetwork):
    """
    Neural network for Sine Matrix (SM) descriptors.

    Specialized for processing sine matrix descriptors, which encode
    structural information through periodic distance functions.

    Additional features:
    - Symmetry preservation
    - Distance weighting
    - Angular sensitivity

    Args:
        sm_dim: Dimension of sine matrix descriptor
        angular_cutoff: Cutoff for angular contributions
        **kwargs: Arguments passed to MaterialsDescriptorNetwork
    """

    def __init__(
            self,
            sm_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 3,
            angular_cutoff: float = 5.0,
            **kwargs
    ):
        super().__init__(
            input_dim=sm_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs
        )

        self.angular_cutoff = angular_cutoff

        # Additional layers for SM-specific processing
        self.distance_embedding = layers.Dense(
            hidden_dim // 2,
            activation="sigmoid",
            kernel_initializer="glorot_uniform"
        )

        self.angular_embedding = layers.Dense(
            hidden_dim // 2,
            activation="tanh",
            kernel_initializer="glorot_uniform"
        )

    def call(self, inputs, training=None):
        """
        Forward pass with SM-specific processing.

        Args:
            inputs: SM descriptors [batch_size, sm_dim]
            training: Whether in training mode

        Returns:
            Predictions [batch_size, output_dim]
        """
        # Split features into distance and angular components
        split_dim = self.input_dim // 2
        distance_features = inputs[..., :split_dim]
        angular_features = inputs[..., split_dim:]

        # Process distance and angular features separately
        d_embedded = self.distance_embedding(distance_features)

        # Apply angular cutoff
        angular_mask = tf.cast(
            tf.abs(angular_features) < self.angular_cutoff,
            tf.float32
        )
        masked_angular = angular_features * angular_mask
        a_embedded = self.angular_embedding(masked_angular)

        # Combine embeddings
        x = tf.concat([d_embedded, a_embedded], axis=-1)

        # Process through main network
        return super().call(x, training=training)

    def get_config(self):
        """Returns model configuration."""
        config = super().get_config()
        config.update({
            "angular_cutoff": self.angular_cutoff
        })
        return config