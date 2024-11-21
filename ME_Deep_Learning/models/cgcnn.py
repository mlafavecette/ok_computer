import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class CGConvLayer(layers.Layer):
    """
    Crystal Graph Convolution Layer

    Implements message passing between atoms in a crystal structure using learnable
    filters based on edge features.

    Args:
        units: Number of output features
        edge_units: Number of edge features
        activation: Activation function
        use_bias: Whether to use bias
        kernel_initializer: Weight initialization method
        batch_norm: Whether to use batch normalization
    """

    def __init__(
            self,
            units: int,
            edge_units: int,
            activation: str = "relu",
            use_bias: bool = True,
            kernel_initializer: str = "glorot_uniform",
            batch_norm: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.edge_units = edge_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.batch_norm = batch_norm

        # Node transformation
        self.node_dense = layers.Dense(
            units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

        # Edge gates
        self.edge_gate = layers.Dense(
            units,
            activation="sigmoid",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

        # Edge filter
        self.edge_filter = layers.Dense(
            units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

        # Optional batch norm
        self.batch_norm_layer = layers.BatchNormalization() if batch_norm else None
        self.activation_fn = layers.Activation(activation)

    def build(self, input_shape):
        """Builds layer weights based on input shape."""
        node_shape, edge_shape, edge_index_shape = input_shape
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass logic.

        Args:
            inputs: Tuple of (node_features, edge_features, edge_indices)
            training: Whether in training mode

        Returns:
            Updated node features after message passing
        """
        node_features, edge_features, edge_indices = inputs

        # Transform node features
        node_hidden = self.node_dense(node_features)

        # Compute edge gates and filters
        edge_hidden = self.edge_gate(edge_features)
        edge_filter = self.edge_filter(edge_features)

        # Gather neighboring node features
        neighbors = tf.gather(node_hidden, edge_indices[:, 1])

        # Apply edge gates and filters
        messages = edge_hidden * edge_filter * neighbors

        # Aggregate messages for each node
        aggregated = tf.math.unsorted_segment_mean(
            messages,
            edge_indices[:, 0],
            num_segments=tf.shape(node_features)[0]
        )

        # Combine with transformed node features
        output = node_hidden + aggregated

        # Apply batch norm and activation if specified
        if self.batch_norm_layer is not None:
            output = self.batch_norm_layer(output, training=training)

        output = self.activation_fn(output)

        return output

    def get_config(self):
        """Returns layer configuration."""
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "edge_units": self.edge_units,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "batch_norm": self.batch_norm,
        }


class Set2SetPooling(layers.Layer):
    """
    Set2Set pooling layer for graph-level outputs.

    Implements memory-based pooling to capture complex interactions between nodes
    as described in the Set2Set paper.

    Args:
        units: Size of memory cell
        processing_steps: Number of processing iterations
        num_layers: Number of LSTM layers
    """

    def __init__(
            self,
            units: int,
            processing_steps: int = 3,
            num_layers: int = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = layers.LSTM(
            units,
            return_state=True,
            num_layers=num_layers
        )

    def build(self, input_shape):
        """Builds layer weights."""
        super().build(input_shape)

    def call(self, inputs, graph_indices):
        """Forward pass logic.

        Args:
            inputs: Node features [num_nodes, feature_dim]
            graph_indices: Graph assignment for each node

        Returns:
            Graph-level features [num_graphs, 2*feature_dim]
        """
        # Initialize memory
        num_graphs = tf.reduce_max(graph_indices) + 1
        h = tf.zeros([num_graphs, self.units])
        c = tf.zeros([num_graphs, self.units])

        # Process for specified number of steps
        for _ in range(self.processing_steps):
            # Get query vector from memory
            q, h, c = self.lstm(h, initial_state=[h, c])

            # Compute attention scores
            e = tf.reduce_sum(inputs * tf.gather(q, graph_indices), axis=-1)
            a = tf.math.segment_softmax(e, graph_indices)

            # Compute read vectors
            r = tf.math.segment_sum(
                tf.expand_dims(a, -1) * inputs,
                graph_indices
            )

            # Update memory
            h = tf.concat([h, r], axis=1)

        return h

    def get_config(self):
        """Returns layer configuration."""
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "processing_steps": self.processing_steps,
            "num_layers": self.num_layers
        }


class CGCNN(Model):
    """
    Crystal Graph Convolutional Neural Network

    Complete CGCNN architecture combining:
    - Pre-processing dense layers
    - CGConv message passing layers
    - Graph-level pooling
    - Post-processing layers

    Args:
        num_features: Number of input node features
        num_edge_features: Number of edge features
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        pre_layers: Number of preprocessing layers
        conv_layers: Number of CGConv layers
        post_layers: Number of post-processing layers
        pool: Pooling method ('mean', 'sum', 'max', 'set2set')
        pool_order: When to apply pooling ('early' or 'late')
        batch_norm: Whether to use batch normalization
        activation: Activation function
        dropout_rate: Dropout rate
    """

    def __init__(
            self,
            num_features: int,
            num_edge_features: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            pre_layers: int = 1,
            conv_layers: int = 3,
            post_layers: int = 1,
            pool: str = "mean",
            pool_order: str = "early",
            batch_norm: bool = True,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pool = pool
        self.pool_order = pool_order

        # Pre-processing layers
        self.pre_layers = []
        for i in range(pre_layers):
            input_dim = num_features if i == 0 else hidden_dim
            self.pre_layers.append(
                layers.Dense(
                    hidden_dim,
                    activation=activation,
                    use_bias=True
                )
            )
            if batch_norm:
                self.pre_layers.append(layers.BatchNormalization())
            if dropout_rate > 0:
                self.pre_layers.append(layers.Dropout(dropout_rate))

        # CGConv layers
        self.conv_layers = []
        for _ in range(conv_layers):
            self.conv_layers.append(
                CGConvLayer(
                    hidden_dim,
                    num_edge_features,
                    activation=activation,
                    batch_norm=batch_norm
                )
            )
            if dropout_rate > 0:
                self.conv_layers.append(layers.Dropout(dropout_rate))

        # Pooling layer
        if pool == "set2set":
            self.pool_layer = Set2SetPooling(hidden_dim)
            pool_dim = 2 * hidden_dim
        else:
            self.pool_layer = None
            pool_dim = hidden_dim

        # Post-processing layers
        self.post_layers = []
        for i in range(post_layers):
            input_dim = pool_dim if i == 0 else hidden_dim
            self.post_layers.append(
                layers.Dense(
                    hidden_dim,
                    activation=activation,
                    use_bias=True
                )
            )
            if batch_norm:
                self.post_layers.append(layers.BatchNormalization())
            if dropout_rate > 0:
                self.post_layers.append(layers.Dropout(dropout_rate))

        # Output layer
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs, training=None):
        """Forward pass logic.

        Args:
            inputs: Tuple of (node_features, edge_features, edge_indices, graph_indices)
            training: Whether in training mode

        Returns:
            Graph-level predictions
        """
        node_features, edge_features, edge_indices, graph_indices = inputs

        # Pre-processing
        x = node_features
        for layer in self.pre_layers:
            x = layer(x, training=training)

        # Message passing
        for conv in self.conv_layers:
            if isinstance(conv, CGConvLayer):
                x = conv((x, edge_features, edge_indices), training=training)
            else:
                x = conv(x, training=training)

        # Pooling
        if self.pool_order == "early":
            if self.pool == "set2set":
                x = self.pool_layer(x, graph_indices)
            elif self.pool == "mean":
                x = tf.math.segment_mean(x, graph_indices)
            elif self.pool == "sum":
                x = tf.math.segment_sum(x, graph_indices)
            elif self.pool == "max":
                x = tf.math.segment_max(x, graph_indices)

        # Post-processing
        for layer in self.post_layers:
            x = layer(x, training=training)

        # Late pooling if specified
        if self.pool_order == "late":
            if self.pool == "set2set":
                x = self.pool_layer(x, graph_indices)
            elif self.pool == "mean":
                x = tf.math.segment_mean(x, graph_indices)
            elif self.pool == "sum":
                x = tf.math.segment_sum(x, graph_indices)
            elif self.pool == "max":
                x = tf.math.segment_max(x, graph_indices)

        # Output
        return self.output_layer(x)

    def get_config(self):
        """Returns model configuration."""
        return {
            "num_features": self.num_features,
            "num_edge_features": self.num_edge_features,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "pre_layers": len(self.pre_layers),
            "conv_layers": len(self.conv_layers),
            "post_layers": len(self.post_layers),
            "pool": self.pool,
            "pool_order": self.pool_order
        }

    @classmethod
    def from_config(cls, config):
        """Creates model from configuration."""
        return cls(**config)