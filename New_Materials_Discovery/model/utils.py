"""
utils.py - Neural Network Primitives and Utilities
Copyright 2024 Cette.ai
Licensed under the Apache License, Version 2.0

This module provides core neural network primitives and utility functions
for materials science applications, optimized for TensorFlow.

Author: Michael R. Lafave
Last Modified: 2021-07-11
"""

import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from functools import partial

# Type aliases
Tensor = Union[tf.Tensor, np.ndarray]
FeaturizerFn = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]

# Constants
EPSILON = 1e-7


@dataclass
class ModelConfig:
    """Configuration for neural network models.

    Attributes:
        hidden_dims: Dimensions of hidden layers
        activation: Activation function name
        use_bias: Whether to use bias terms
        dropout_rate: Dropout rate
        weight_init_std: Weight initialization std
        normalization: Type of normalization to use
    """
    hidden_dims: Tuple[int, ...] = (128, 64)
    activation: str = "swish"
    use_bias: bool = True
    dropout_rate: float = 0.0
    weight_init_std: Optional[float] = None
    normalization: str = "layer_norm"


class BetaSwish(tf.keras.layers.Layer):
    """Swish activation with learnable beta parameter."""

    def build(self, input_shape: tf.TensorShape):
        """Build layer, creating beta parameter."""
        features = input_shape[-1]
        self.beta = self.add_weight(
            "beta",
            shape=(features,),
            initializer="ones",
            trainable=True
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass of Beta-Swish activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return x * tf.sigmoid(self.beta * x)


# Dictionary of available activation functions
NONLINEARITY = {
    "none": lambda x: x,
    "relu": tf.nn.relu,
    "swish": BetaSwish(),
    "raw_swish": tf.nn.swish,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid,
    "silu": tf.nn.silu,
}


def get_nonlinearity(name: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """Get activation function by name.

    Args:
        name: Name of activation function

    Returns:
        Activation function

    Raises:
        ValueError: If activation not found
    """
    if name in NONLINEARITY:
        return NONLINEARITY[name]
    raise ValueError(f"Nonlinearity '{name}' not found")


class DenseLayer(tf.keras.layers.Layer):
    """Dense layer with configurable initialization and normalization."""

    def __init__(
            self,
            units: int,
            activation: Optional[str] = None,
            use_bias: bool = True,
            kernel_std: Optional[float] = None,
            normalization: Optional[str] = None,
            dropout_rate: float = 0.0,
            name: str = "dense"
    ):
        """Initialize dense layer.

        Args:
            units: Number of output units
            activation: Activation function name
            use_bias: Whether to use bias
            kernel_std: Standard deviation for kernel init
            normalization: Type of normalization
            dropout_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        self.units = units
        self.activation = get_nonlinearity(activation) if activation else None
        self.use_bias = use_bias
        self.kernel_std = kernel_std
        self.normalization = normalization
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        """Build layer components."""
        # Kernel initialization
        if self.kernel_std is not None:
            kernel_init = tf.keras.initializers.RandomNormal(stddev=self.kernel_std)
        else:
            kernel_init = "glorot_uniform"

        # Create weights
        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1], self.units],
            initializer=kernel_init
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[self.units],
                initializer="zeros"
            )

        # Normalization layer
        if self.normalization == "layer_norm":
            self.norm = tf.keras.layers.LayerNormalization()
        elif self.normalization == "batch_norm":
            self.norm = tf.keras.layers.BatchNormalization()
        else:
            self.norm = None

        # Dropout layer
        if self.dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(
            self,
            inputs: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        """Forward pass of dense layer.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        x = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.norm is not None:
            x = self.norm(x, training=training)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout_rate > 0 and training:
            x = self.dropout(x, training=training)

        return x


class MLP(tf.keras.layers.Layer):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(
            self,
            hidden_dims: Tuple[int, ...],
            activation: str = "swish",
            output_activation: Optional[str] = None,
            use_bias: bool = True,
            dropout_rate: float = 0.0,
            normalization: Optional[str] = None,
            name: str = "mlp"
    ):
        """Initialize MLP.

        Args:
            hidden_dims: Hidden layer dimensions
            activation: Hidden layer activation
            output_activation: Output activation
            use_bias: Whether to use bias
            dropout_rate: Dropout rate
            normalization: Type of normalization
            name: Layer name
        """
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.normalization = normalization

    def build(self, input_shape: tf.TensorShape):
        """Build MLP layers."""
        # Hidden layers
        self.hidden_layers = []
        for i, units in enumerate(self.hidden_dims[:-1]):
            self.hidden_layers.append(
                DenseLayer(
                    units=units,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    normalization=self.normalization,
                    dropout_rate=self.dropout_rate,
                    name=f"dense_{i}"
                )
            )

        # Output layer
        self.output_layer = DenseLayer(
            units=self.hidden_dims[-1],
            activation=self.output_activation,
            use_bias=self.use_bias,
            name="output"
        )

    def call(
            self,
            inputs: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        """Forward pass of MLP.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x, training=training)


class BesselEmbedding(tf.keras.layers.Layer):
    """Bessel function embedding for distance features."""

    def __init__(
            self,
            num_basis: int,
            inner_cutoff: float,
            outer_cutoff: float,
            trainable: bool = False,
            name: str = "bessel_embedding"
    ):
        """Initialize Bessel embedding.

        Args:
            num_basis: Number of basis functions
            inner_cutoff: Inner cutoff radius
            outer_cutoff: Outer cutoff radius
            trainable: Whether frequencies are trainable
            name: Layer name
        """
        super().__init__(name=name)
        self.num_basis = num_basis
        self.inner_cutoff = inner_cutoff
        self.outer_cutoff = outer_cutoff
        self.trainable = trainable

    def build(self, input_shape: tf.TensorShape):
        """Build layer, initializing frequencies."""
        # Initialize as n*pi
        frequencies = np.pi * np.arange(1, self.num_basis + 1)

        self.frequencies = self.add_weight(
            "frequencies",
            shape=[self.num_basis],
            initializer=tf.constant_initializer(frequencies),
            trainable=self.trainable
        )

    def call(self, r: tf.Tensor) -> tf.Tensor:
        """Compute Bessel embeddings.

        Args:
            r: Distance tensor

        Returns:
            Bessel function values
        """
        # Safe handling of small r
        r_safe = tf.where(r > 1e-5, r, tf.ones_like(r) * 1e3)

        # Compute Bessel functions
        x = tf.einsum('b,n->bn', r_safe, self.frequencies)
        bessel = 2.0 / self.outer_cutoff * tf.sin(x / self.outer_cutoff) / r_safe[..., None]

        # Apply smooth cutoff
        cutoff = smooth_cutoff(r, self.inner_cutoff, self.outer_cutoff)
        return tf.where(r[..., None] > 1e-5, bessel * cutoff[..., None], 0.0)


def smooth_cutoff(
        r: tf.Tensor,
        r_in: float,
        r_out: float
) -> tf.Tensor:
    """Smooth cutoff function.

    Args:
        r: Distance tensor
        r_in: Inner cutoff
        r_out: Outer cutoff

    Returns:
        Smooth cutoff values
    """
    x = (r - r_in) / (r_out - r_in)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return 1.0 - 6.0 * x ** 5 + 15.0 * x ** 4 - 10.0 * x ** 3


# Dataset statistics for normalization
DATASET_STATS = {
    "harder_silicon": {
        "shift": 2.2548,
        "scale": 0.8825
    }
}


def get_normalization(
        config: Dict[str, Any]
) -> Tuple[float, float]:
    """Get normalization parameters.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (shift, scale)

    Raises:
        ValueError: If normalization not found
    """
    if "shift" in config and "scale" in config:
        return config["shift"], config["scale"]
    elif "dataset" in config:
        stats = DATASET_STATS.get(config["dataset"])
        if stats:
            return stats["shift"], stats["scale"]
    raise ValueError("Normalization parameters not found")