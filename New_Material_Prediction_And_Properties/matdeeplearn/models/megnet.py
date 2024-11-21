import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class EdgeModel(layers.Layer):
    """Edge update model for MEGNet.

    This layer updates edge features using information from connected nodes
    and global state.
    """

    def __init__(self,
                 hidden_dim: int,
                 activation: str = 'softplus',
                 **kwargs):
        super(EdgeModel, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim

        # Edge update network
        self.edge_mlp = Sequential([
            layers.Dense(hidden_dim, activation=activation),
            layers.Dense(hidden_dim, activation=activation),
            layers.Dense(hidden_dim)
        ])

    def build(self, input_shape):
        # Dense layer for combining all inputs
        self.combine = layers.Dense(self.hidden_dim)
        super(EdgeModel, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Update edge features.

        Args:
            inputs: Tuple of (edge_attr, src_features, dst_features, global_features)

        Returns:
            Updated edge features
        """
        edge_attr, src, dst, global_feat = inputs

        # Combine all inputs
        src_expand = tf.repeat(src, tf.shape(edge_attr)[0] // tf.shape(src)[0], axis=0)
        dst_expand = tf.repeat(dst, tf.shape(edge_attr)[0] // tf.shape(dst)[0], axis=0)
        global_expand = tf.repeat(global_feat,
                                  tf.shape(edge_attr)[0] // tf.shape(global_feat)[0],
                                  axis=0)

        inputs_concat = tf.concat([edge_attr, src_expand, dst_expand, global_expand],
                                  axis=-1)
        combined = self.combine(inputs_concat)

        # Process through MLP
        return self.edge_mlp(combined)


class NodeModel(layers.Layer):
    """Node update model for MEGNet.

    This layer updates node features using aggregated edge information
    and global state.
    """

    def __init__(self,
                 hidden_dim: int,
                 activation: str = 'softplus',
                 **kwargs):
        super(NodeModel, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim

        # Node update network
        self.node_mlp = Sequential([
            layers.Dense(hidden_dim, activation=activation),
            layers.Dense(hidden_dim, activation=activation),
            layers.Dense(hidden_dim)
        ])

    def build(self, input_shape):
        self.combine = layers.Dense(self.hidden_dim)
        super(NodeModel, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Update node features.

        Args:
            inputs: Tuple of (x, edge_index, edge_features, global_features)

        Returns:
            Updated node features
        """
        x, edge_index, edge_features, global_feat = inputs

        # Aggregate edge features
        row, col = edge_index[0], edge_index[1]
        edge_aggr = tf.scatter_nd(
            tf.expand_dims(row, 1),
            edge_features,
            shape=[tf.shape(x)[0], self.hidden_dim]
        )

        # Combine with global features
        global_expand = tf.repeat(global_feat, tf.shape(x)[0] // tf.shape(global_feat)[0], axis=0)
        combined = self.combine(tf.concat([x, edge_aggr, global_expand], axis=-1))

        # Process through MLP
        return self.node_mlp(combined)


class GlobalModel(layers.Layer):
    """Global state update model for MEGNet.

    This layer updates global features using aggregated information
    from nodes and edges.
    """

    def __init__(self,
                 hidden_dim: int,
                 activation: str = 'softplus',
                 **kwargs):
        super(GlobalModel, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim

        # Global update network
        self.global_mlp = Sequential([
            layers.Dense(hidden_dim, activation=activation),
            layers.Dense(hidden_dim, activation=activation),
            layers.Dense(hidden_dim)
        ])

    def build(self, input_shape):
        self.combine = layers.Dense(self.hidden_dim)
        super(GlobalModel, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
        """Update global features.

                Args:
                    inputs: Tuple of (global_features, node_features, edge_features, batch_idx)

                Returns:
                    Updated global features
                """
        global_feat, node_features, edge_features, batch_idx = inputs

        # Aggregate node and edge features per graph
        node_aggr = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            node_features,
            shape=[tf.shape(global_feat)[0], self.hidden_dim]
        )

        edge_aggr = tf.scatter_nd(
            tf.expand_dims(batch_idx, 1),
            edge_features,
            shape=[tf.shape(global_feat)[0], self.hidden_dim]
        )

        # Combine all features
        combined = self.combine(tf.concat([global_feat, node_aggr, edge_aggr], axis=-1))

        # Process through MLP
        return self.global_mlp(combined)

    class MEGNetBlock(layers.Layer):
        """MEGNet interaction block combining edge, node, and global updates.

        This block implements a full message passing step in MEGNet, coordinating
        updates across all three levels of the graph representation.
        """

        def __init__(self,
                     hidden_dim: int,
                     activation: str = 'softplus',
                     **kwargs):
            super(MEGNetBlock, self).__init__(**kwargs)

            self.edge_model = EdgeModel(hidden_dim, activation)
            self.node_model = NodeModel(hidden_dim, activation)
            self.global_model = GlobalModel(hidden_dim, activation)

        def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            """Process one complete message passing step.

            Args:
                inputs: Tuple of (node_features, edge_index, edge_features, global_features, batch_idx)

            Returns:
                Tuple of updated (node_features, edge_features, global_features)
            """
            x, edge_index, edge_attr, global_feat, batch_idx = inputs
            row, col = edge_index[0], edge_index[1]

            # Edge updates
            src_features = tf.gather(x, row)
            dst_features = tf.gather(x, col)
            edge_attr_updated = self.edge_model((edge_attr, src_features, dst_features, global_feat))

            # Node updates
            x_updated = self.node_model((x, edge_index, edge_attr_updated, global_feat))

            # Global updates
            global_feat_updated = self.global_model((global_feat, x_updated, edge_attr_updated, batch_idx))

            return x_updated, edge_attr_updated, global_feat_updated

    class MEGNet(Model):
        """Materials Exploration Graph Neural Network.

        This model implements the complete MEGNet architecture for materials property prediction,
        featuring multi-level message passing between nodes, edges, and global state.

        Key Features:
        - Three-level information processing (nodes, edges, global)
        - Multiple MEGNet blocks for deep hierarchical learning
        - Sophisticated state tracking and updates
        - Physics-informed architecture design
        - Flexible output prediction for various materials properties
        """

        def __init__(self,
                     node_features: int,
                     edge_features: int,
                     global_features: int,
                     hidden_dim: int = 64,
                     num_blocks: int = 3,
                     num_dense_layers: int = 2,
                     output_dim: int = 1,
                     activation: str = 'softplus',
                     dropout_rate: float = 0.0,
                     **kwargs):
            super(MEGNet, self).__init__(**kwargs)

            self.hidden_dim = hidden_dim
            self.output_dim = output_dim

            # Initial embeddings
            self.node_embedding = Sequential([
                layers.Dense(hidden_dim),
                layers.Activation(activation)
            ])

            self.edge_embedding = Sequential([
                layers.Dense(hidden_dim),
                layers.Activation(activation)
            ])

            self.global_embedding = Sequential([
                layers.Dense(hidden_dim),
                layers.Activation(activation)
            ])

            # MEGNet blocks
            self.megnet_blocks = [
                MEGNetBlock(hidden_dim, activation)
                for _ in range(num_blocks)
            ]

            # Output network
            output_layers = []
            for _ in range(num_dense_layers):
                output_layers.extend([
                    layers.Dense(hidden_dim),
                    layers.Activation(activation),
                    layers.Dropout(dropout_rate)
                ])
            output_layers.append(layers.Dense(output_dim))
            self.output_network = Sequential(output_layers)

        def call(self, inputs: Tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor:
            """Forward pass of the model.

            Args:
                inputs: Tuple of (node_attr, edge_index, edge_attr, global_attr, batch_idx)
                training: Whether in training mode

            Returns:
                Predicted properties [batch_size, output_dim]
            """
            x, edge_index, edge_attr, global_attr, batch_idx = inputs

            # Initial embeddings
            x = self.node_embedding(x)
            edge_attr = self.edge_embedding(edge_attr)
            global_attr = self.global_embedding(global_attr)

            # Process through MEGNet blocks
            for block in self.megnet_blocks:
                x_res, edge_attr_res, global_attr_res = block(
                    (x, edge_index, edge_attr, global_attr, batch_idx),
                    training=training
                )
                # Residual connections
                x = x + x_res
                edge_attr = edge_attr + edge_attr_res
                global_attr = global_attr + global_attr_res

            # Pool node features to graph level
            num_graphs = tf.reduce_max(batch_idx) + 1
            node_pool = tf.scatter_nd(
                tf.expand_dims(batch_idx, 1),
                x,
                shape=[num_graphs, self.hidden_dim]
            )

            # Combine with global features
            final_features = tf.concat([node_pool, global_attr], axis=-1)

            # Final prediction
            out = self.output_network(final_features, training=training)
            return out if self.output_dim > 1 else tf.squeeze(out, -1)

        def compute_loss(self, data, training: bool = False) -> Tuple[tf.Tensor, dict]:
            """Compute model loss with additional performance metrics.

            Args:
                data: Tuple of (inputs, targets)
                training: Whether in training mode

            Returns:
                Tuple of (loss, metrics_dict)
            """
            inputs, targets = data

            with tf.GradientTape() as tape:
                predictions = self(inputs, training=training)

                # Compute main loss
                main_loss = tf.reduce_mean(tf.square(predictions - targets))

                # Add regularization if needed
                if self.losses:
                    reg_loss = tf.add_n(self.losses)
                    total_loss = main_loss + reg_loss
                else:
                    total_loss = main_loss

            if training:
                # Compute and apply gradients
                gradients = tape.gradient(total_loss, self.trainable_variables)
                gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                metrics = {
                    'loss': total_loss,
                    'main_loss': main_loss,
                    'gradient_norm': grad_norm
                }

                if self.losses:
                    metrics['reg_loss'] = reg_loss

                return total_loss, metrics

            return total_loss, {'loss': total_loss}