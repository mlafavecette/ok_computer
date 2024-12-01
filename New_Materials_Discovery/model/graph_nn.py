"""
graph_network.py - Base Graph Neural Network Implementation
Copyright 2024 Cette
Licensed under the Apache License, Version 2.0

This module implements the core graph neural network functionality,
providing base classes and utilities for building graph-based models.

Author: Michael Lafave
Last Modified: 2022-11-23
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass


@dataclass
class GraphData:
    """Container for graph structured data.

    Attributes:
        nodes: Node features
        edges: Edge features
        senders: Source node indices for edges
        receivers: Target node indices for edges
        globals: Global graph features
        n_node: Number of nodes per graph
        n_edge: Number of edges per graph
    """
    nodes: tf.Tensor
    edges: tf.Tensor
    senders: tf.Tensor
    receivers: tf.Tensor
    globals: Optional[tf.Tensor] = None
    n_node: Optional[tf.Tensor] = None
    n_edge: Optional[tf.Tensor] = None


class GraphNetwork(tf.keras.layers.Layer):
    """Base class for graph neural networks."""

    def __init__(
            self,
            update_node_fn: Optional[Callable] = None,
            update_edge_fn: Optional[Callable] = None,
            update_global_fn: Optional[Callable] = None,
            aggregate_edges_for_nodes_fn: Callable = tf.math.segment_sum,
            aggregate_nodes_for_globals_fn: Callable = tf.math.segment_mean,
            aggregate_edges_for_globals_fn: Callable = tf.math.segment_mean,
            name: str = "graph_network"
    ):
        """Initialize graph network.

        Args:
            update_node_fn: Node update function
            update_edge_fn: Edge update function
            update_global_fn: Global state update function
            aggregate_edges_for_nodes_fn: Edge aggregation for nodes
            aggregate_nodes_for_globals_fn: Node aggregation for globals
            aggregate_edges_for_globals_fn: Edge aggregation for globals
            name: Layer name
        """
        super().__init__(name=name)
        self.update_node_fn = update_node_fn
        self.update_edge_fn = update_edge_fn
        self.update_global_fn = update_global_fn
        self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn

    def call(self, graph: GraphData) -> GraphData:
        """Forward pass of graph network.

        Args:
            graph: Input graph data

        Returns:
            Updated graph data
        """
        # Extract components
        nodes = graph.nodes
        edges = graph.edges
        globals_ = graph.globals
        senders = graph.senders
        receivers = graph.receivers

        # Update edges if specified
        if self.update_edge_fn is not None:
            sender_features = tf.gather(nodes, senders)
            receiver_features = tf.gather(nodes, receivers)
            edges = self.update_edge_fn(edges, sender_features, receiver_features)

        # Update nodes if specified
        if self.update_node_fn is not None:
            # Aggregate edge features
            aggregated_edges = self.aggregate_edges_for_nodes_fn(
                edges,
                receivers,
                num_segments=tf.shape(nodes)[0]
            )
            nodes = self.update_node_fn(nodes, aggregated_edges)

        # Update global features if specified
        if self.update_global_fn is not None and globals_ is not None:
            # Aggregate node and edge features
            aggregated_nodes = self.aggregate_nodes_for_globals_fn(
                nodes,
                graph.n_node,
                num_segments=tf.shape(globals_)[0]
            )
            aggregated_edges = self.aggregate_edges_for_globals_fn(
                edges,
                graph.n_edge,
                num_segments=tf.shape(globals_)[0]
            )
            globals_ = self.update_global_fn(
                globals_,
                aggregated_nodes,
                aggregated_edges
            )

        return GraphData(
            nodes=nodes,
            edges=edges,
            globals=globals_,
            senders=senders,
            receivers=receivers,
            n_node=graph.n_node,
            n_edge=graph.n_edge
        )


class GraphAttention(tf.keras.layers.Layer):
    """Continued from above..."""

    def __init__(
            self,
            num_heads: int = 4,
            key_dim: int = 32,
            dropout: float = 0.1,
            name: str = "graph_attention"
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout

    def build(self, input_shape: tf.TensorShape):
        """Build attention layer components."""
        hidden_dim = input_shape[-1]

        # Create attention projections
        self.query_proj = tf.keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=False
        )
        self.key_proj = tf.keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=False
        )
        self.value_proj = tf.keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=False
        )

        # Output projection
        self.output_proj = tf.keras.layers.Dense(hidden_dim)

    def call(
            self,
            nodes: tf.Tensor,
            edges: tf.Tensor,
            senders: tf.Tensor,
            receivers: tf.Tensor,
            mask: Optional[tf.Tensor] = None,
            training: bool = False
    ) -> tf.Tensor:
        """Forward pass of graph attention.

        Args:
            nodes: Node features
            edges: Edge features
            senders: Sender node indices
            receivers: Receiver node indices
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Updated node features
        """
        # Gather features
        sender_features = tf.gather(nodes, senders)
        receiver_features = tf.gather(nodes, receivers)

        # Compute queries, keys and values
        q = self._reshape_heads(self.query_proj(receiver_features))
        k = self._reshape_heads(self.key_proj(sender_features))
        v = self._reshape_heads(self.value_proj(sender_features))

        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))

        # Apply mask if provided
        if mask is not None:
            scores += (1.0 - tf.cast(mask, scores.dtype)) * -1e9

        # Apply attention
        attn_weights = tf.nn.softmax(scores, axis=-1)
        if training:
            attn_weights = tf.nn.dropout(attn_weights, self.dropout)

        # Compute attention output
        output = tf.matmul(attn_weights, v)
        output = self._reshape_from_heads(output)

        # Final projection
        return self.output_proj(output)

    def _reshape_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Reshape input tensor for multi-head attention."""
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))

    def _reshape_from_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Reshape tensor back from multi-head format."""
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, (batch_size, -1, self.num_heads * self.key_dim))


class MessagePassingNetwork(GraphNetwork):
    """Implements message passing neural network architecture."""

    def __init__(
            self,
            hidden_dim: int = 64,
            num_layers: int = 3,
            use_attention: bool = True,
            dropout: float = 0.1,
            name: str = "message_passing"
    ):
        """Initialize message passing network.

        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of message passing layers
            use_attention: Whether to use attention
            dropout: Dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.dropout = dropout

    def build(self, input_shape: tf.TensorShape):
        """Build network components."""
        # Create message passing layers
        self.message_layers = []
        for i in range(self.num_layers):
            if self.use_attention:
                message_fn = GraphAttention(
                    num_heads=4,
                    key_dim=self.hidden_dim // 4,
                    dropout=self.dropout,
                    name=f"attention_{i}"
                )
            else:
                message_fn = tf.keras.Sequential([
                    tf.keras.layers.Dense(
                        self.hidden_dim,
                        activation="swish"
                    ),
                    tf.keras.layers.Dropout(self.dropout),
                    tf.keras.layers.Dense(self.hidden_dim)
                ], name=f"message_{i}")

            update_fn = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    self.hidden_dim,
                    activation="swish"
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(self.hidden_dim)
            ], name=f"update_{i}")

            self.message_layers.append((message_fn, update_fn))

    def call(
            self,
            graph: GraphData,
            training: bool = False
    ) -> GraphData:
        """Forward pass of message passing network.

        Args:
            graph: Input graph data
            training: Whether in training mode

        Returns:
            Updated graph data
        """
        nodes = graph.nodes
        edges = graph.edges

        # Message passing layers
        for message_fn, update_fn in self.message_layers:
            # Compute messages
            if self.use_attention:
                messages = message_fn(
                    nodes,
                    edges,
                    graph.senders,
                    graph.receivers,
                    training=training
                )
            else:
                sender_features = tf.gather(nodes, graph.senders)
                receiver_features = tf.gather(nodes, graph.receivers)
                messages = message_fn(
                    tf.concat([sender_features, receiver_features, edges], axis=-1)
                )

            # Aggregate messages
            aggregated = tf.math.segment_sum(
                messages,
                graph.receivers,
                num_segments=tf.shape(nodes)[0]
            )

            # Update node features
            nodes = update_fn(tf.concat([nodes, aggregated], axis=-1))

            if training:
                nodes = tf.nn.dropout(nodes, self.dropout)

        return GraphData(
            nodes=nodes,
            edges=graph.edges,
            senders=graph.senders,
            receivers=graph.receivers,
            globals=graph.globals,
            n_node=graph.n_node,
            n_edge=graph.n_edge
        )


def create_graph_network(
        network_type: str = "message_passing",
        **kwargs
) -> GraphNetwork:
    """Factory function to create graph networks.

    Args:
        network_type: Type of graph network to create
        **kwargs: Additional arguments for network

    Returns:
        Initialized graph network
    """
    if network_type == "message_passing":
        return MessagePassingNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


class ResidualGraphNetwork(GraphNetwork):
    """Graph network with residual connections."""

    def __init__(
            self,
            hidden_dim: int = 64,
            num_layers: int = 3,
            dropout: float = 0.1,
            name: str = "residual_graph_network"
    ):
        """Initialize residual graph network.

        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            dropout: Dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Create layers
        self.layers = [
            MessagePassingNetwork(
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=dropout,
                name=f"layer_{i}"
            )
            for i in range(num_layers)
        ]

        # Layer normalization
        self.layer_norms = [
            tf.keras.layers.LayerNormalization()
            for _ in range(num_layers)
        ]

    def call(
            self,
            graph: GraphData,
            training: bool = False
    ) -> GraphData:
        """Forward pass with residual connections.

        Args:
            graph: Input graph data
            training: Whether in training mode

        Returns:
            Updated graph data
        """
        x = graph

        for layer, norm in zip(self.layers, self.layer_norms):
            # Save residual
            residual = x.nodes

            # Apply layer
            x = layer(x, training=training)

            # Add residual and normalize
            x = GraphData(
                nodes=norm(residual + x.nodes),
                edges=x.edges,
                senders=x.senders,
                receivers=x.receivers,
                globals=x.globals,
                n_node=x.n_node,
                n_edge=x.n_edge
            )

        return x