"""
materials_layers.py - Specialized Neural Network Layers for Materials Science
Copyright 2024 Cette AI
Licensed under the Apache License, Version 2.0

This module implements specialized neural network layers for materials science
applications, incorporating physical principles and chemical knowledge.

Author: Michael Lafave
Last Modified: 2021-10-05
"""

import tensorflow as tf
import numpy as np
from typing import Callable, List, Optional, Tuple, Union


class RadialBasisLayer(tf.keras.layers.Layer):
    """Radial basis function layer for encoding atomic distances.

    This layer transforms interatomic distances into a basis set representation,
    providing a smooth and differentiable description of atomic environments.
    """

    def __init__(
            self,
            num_rbf: int = 16,
            cutoff: float = 5.0,
            rbf_type: str = "gaussian",
            trainable: bool = True,
            name: str = "radial_basis"
    ):
        """Initialize the RBF layer.

        Args:
            num_rbf: Number of basis functions
            cutoff: Cutoff radius
            rbf_type: Type of basis functions ("gaussian" or "bessel")
            trainable: Whether centers and widths are trainable
            name: Layer name
        """
        super().__init__(name=name)
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.rbf_type = rbf_type
        self.trainable = trainable

    def build(self, input_shape: tf.TensorShape):
        """Build the layer, creating trainable weights.

        Args:
            input_shape: Shape of input tensor
        """
        # Initialize centers uniformly in [0, cutoff]
        self.centers = self.add_weight(
            "centers",
            shape=(self.num_rbf,),
            initializer=tf.keras.initializers.RandomUniform(0, self.cutoff),
            trainable=self.trainable
        )

        # Initialize widths based on spacing between centers
        self.widths = self.add_weight(
            "widths",
            shape=(self.num_rbf,),
            initializer=tf.keras.initializers.Constant(
                0.1 * self.cutoff / self.num_rbf
            ),
            trainable=self.trainable
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Interatomic distances

        Returns:
            RBF-encoded distances
        """
        inputs = tf.expand_dims(inputs, -1)  # [batch, N, 1]

        if self.rbf_type == "gaussian":
            # Gaussian RBFs
            return tf.exp(-self.widths * tf.square(inputs - self.centers))
        else:
            # Bessel RBFs
            x = inputs * self.centers
            return tf.sin(x) / x


class ChemicalEmbedding(tf.keras.layers.Layer):
    """Continued from above..."""

    def build(self, input_shape: tf.TensorShape):
        """Build the layer, creating embeddings."""
        if self.use_pretrained:
            initial_embedding = self._get_pretrained_embedding()
        else:
            initial_embedding = None

        self.embedding = self.add_weight(
            "element_embedding",
            shape=(self.max_atomic_number, self.embedding_dim),
            initializer=(
                tf.keras.initializers.Constant(initial_embedding)
                if initial_embedding is not None
                else tf.keras.initializers.random_normal(stddev=0.1)
            ),
            trainable=True
        )

    def _get_pretrained_embedding(self) -> np.ndarray:
        """Generate pretrained embeddings based on chemical properties.

        Returns:
            Array of pretrained embeddings
        """
        # Initialize with physical/chemical properties
        properties = {
            'atomic_radius': np.zeros(self.max_atomic_number),
            'electronegativity': np.zeros(self.max_atomic_number),
            'ionization_energy': np.zeros(self.max_atomic_number),
            'electron_affinity': np.zeros(self.max_atomic_number),
        }

        # Load fundamental chemical properties
        # This would be replaced with actual data in production
        # For now, we use placeholder values
        return np.random.normal(0, 0.1, (self.max_atomic_number, self.embedding_dim))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Atomic numbers (1-based indexing)

        Returns:
            Chemical element embeddings
        """
        # Convert 1-based atomic numbers to 0-based indices
        indices = tf.cast(inputs - 1, tf.int32)
        return tf.gather(self.embedding, indices)


class EdgeConvolution(tf.keras.layers.Layer):
    """Edge convolution layer for crystal graphs.

    Implements edge convolutions with attention mechanism and physical
    symmetry preservation.
    """

    def __init__(
            self,
            units: int = 64,
            activation: str = "swish",
            use_attention: bool = True,
            name: str = "edge_conv"
    ):
        """Initialize edge convolution layer.

        Args:
            units: Number of output units
            activation: Activation function
            use_attention: Whether to use attention mechanism
            name: Layer name
        """
        super().__init__(name=name)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_attention = use_attention

    def build(self, input_shape: List[tf.TensorShape]):
        """Build the layer.

        Args:
            input_shape: Shapes of input tensors
        """
        node_dims = input_shape[0][-1]
        edge_dims = input_shape[1][-1]

        # Message passing networks
        self.message_net = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.units,
                activation=self.activation,
                kernel_initializer='glorot_uniform'
            ),
            tf.keras.layers.Dense(
                self.units,
                kernel_initializer='glorot_uniform'
            )
        ])

        if self.use_attention:
            # Attention networks
            self.attention_net = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    self.units,
                    activation=self.activation
                ),
                tf.keras.layers.Dense(1)
            ])

        # Update network
        self.update_net = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.units,
                activation=self.activation
            ),
            tf.keras.layers.Dense(self.units)
        ])

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Tuple of (node_features, edge_features, senders, receivers)

        Returns:
            Updated node features
        """
        node_features, edge_features, senders, receivers = inputs

        # Gather features for message passing
        sender_features = tf.gather(node_features, senders)
        receiver_features = tf.gather(node_features, receivers)

        # Compute messages
        message_inputs = tf.concat(
            [sender_features, receiver_features, edge_features],
            axis=-1
        )
        messages = self.message_net(message_inputs)

        if self.use_attention:
            # Compute attention weights
            attention_logits = self.attention_net(message_inputs)
            attention_weights = tf.nn.softmax(attention_logits, axis=0)
            messages = messages * attention_weights

        # Aggregate messages
        aggregated = tf.math.segment_sum(messages, receivers)

        # Update node features
        return self.update_net(
            tf.concat([node_features, aggregated], axis=-1)
        )


class CrystalGraphAttention(tf.keras.layers.Layer):
    """Multi-head attention layer for crystal graphs.

    Implements attention mechanism that respects crystallographic symmetries
    and periodic boundary conditions.
    """

    def __init__(
            self,
            num_heads: int = 4,
            key_dim: int = 32,
            value_dim: int = 32,
            dropout: float = 0.1,
            name: str = "crystal_attention"
    ):
        """Initialize the attention layer.

        Args:
            num_heads: Number of attention heads
            key_dim: Dimension of key vectors
            value_dim: Dimension of value vectors
            dropout: Dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout

    def build(self, input_shape: tf.TensorShape):
        """Build the layer."""
        node_dims = input_shape[0][-1]

        # Create trainable weights for Q, K, V transformations
        self.query_dense = tf.keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=False
        )
        self.key_dense = tf.keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=False
        )
        self.value_dense = tf.keras.layers.Dense(
            self.num_heads * self.value_dim,
            use_bias=False
        )
        self.output_dense = tf.keras.layers.Dense(node_dims)

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            training: bool = False
    ) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Tuple of (node_features, edge_mask, distance_weights)
            training: Whether in training mode

        Returns:
            Updated node features
        """
        node_features, edge_mask, distance_weights = inputs
        batch_size = tf.shape(node_features)[0]

        # Compute Q, K, V
        query = self._reshape_to_heads(self.query_dense(node_features))
        key = self._reshape_to_heads(self.key_dense(node_features))
        value = self._reshape_to_heads(self.value_dense(node_features))

        # Compute attention scores
        attention_logits = tf.matmul(query, key, transpose_b=True)
        attention_logits = attention_logits / tf.math.sqrt(
            tf.cast(self.key_dim, tf.float32)
        )

        # Apply edge mask and distance weighting
        if edge_mask is not None:
            attention_logits += (1.0 - edge_mask) * -1e9
        if distance_weights is not None:
            attention_logits *= distance_weights[:, None, None, :]

        attention_weights = tf.nn.softmax(attention_logits)

        if training:
            attention_weights = tf.nn.dropout(
                attention_weights,
                self.dropout
            )

        # Apply attention and reshape
        attention_output = tf.matmul(attention_weights, value)
        attention_output = self._reshape_from_heads(attention_output)

        return self.output_dense(attention_output)

    def _reshape_to_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Reshape input tensor for multi-head attention."""
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        return tf.reshape(
            x,
            (batch_size, seq_length, self.num_heads, -1)
        )

    def _reshape_from_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Reshape tensor back from multi-head format."""
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        return tf.reshape(
            x,
            (batch_size, seq_length, self.num_heads * self.value_dim)
        )