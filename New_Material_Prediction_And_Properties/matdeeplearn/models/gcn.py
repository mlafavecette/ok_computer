import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from typing import List, Tuple, Union, Optional


class GCNConvolution(layers.Layer):
    """Graph Convolutional Network layer.

    Implements the GCN convolution operation with optional edge weights
    and improved normalization scheme.
    """

    def __init__(self,
                 output_dim: int,
                 improved: bool = True,
                 add_self_loops: bool = False,
                 **kwargs):
        super(GCNConvolution, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.improved = improved
        self.add_self_loops = add_self_loops

        # Trainable weight matrix
        self.weight = self.add_weight(
            shape=(None, output_dim),  # Input dim set in build()
            initializer='glorot_uniform',
            trainable=True,
            name='weight'
        )

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.weight.set_shape((input_dim, self.output_dim))
        super(GCNConvolution, self).build(input_shape)

    def compute_normalization(self,
                              edge_index: tf.Tensor,
                              num_nodes: int,
                              edge_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Compute GCN normalization factor.

        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Number of nodes in the graph
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Normalized edge weights
        """
        row, col = edge_index[0], edge_index[1]

        if edge_weight is None:
            edge_weight = tf.ones_like(row, dtype=tf.float32)

        # Calculate degree for normalization
        degree = tf.scatter_nd(
            tf.expand_dims(row, 1),
            edge_weight,
            shape=[num_nodes]
        )

        if self.improved:
            degree = degree + 1.0

        deg_inv_sqrt = tf.pow(degree, -0.5)
        deg_inv_sqrt = tf.where(
            tf.math.is_inf(deg_inv_sqrt),
            tf.zeros_like(deg_inv_sqrt),
            deg_inv_sqrt
        )

        # Compute normalization
        norm = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

        return norm

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Tuple of (x, edge_index, edge_weight)

        Returns:
            Updated node features
        """
        x, edge_index, edge_weight = inputs
        num_nodes = tf.shape(x)[0]

        # Linear transformation
        out = tf.matmul(x, self.weight)

        # Add self-loops if specified
        if self.add_self_loops:
            loop_index = tf.range(num_nodes)
            loop_index = tf.stack([loop_index, loop_index])
            edge_index = tf.concat([edge_index, loop_index], axis=1)
            if edge_weight is not None:
                loop_weight = tf.ones(num_nodes)
                edge_weight = tf.concat([edge_weight, loop_weight], axis=0)

                # Compute normalization
            norm = self.compute_normalization(edge_index, num_nodes, edge_weight)

            # Message passing
            row, col = edge_index[0], edge_index[1]
            messages = tf.gather(out, col) * tf.expand_dims(norm, -1)

            # Aggregate messages
            out = tf.scatter_nd(
                tf.expand_dims(row, 1),
                messages,
                shape=[num_nodes, self.output_dim]
            )

            return out

        class GCN(Model):
            """Graph Convolutional Network for crystal property prediction.

            This model implements a standard GCN architecture with optional batch normalization,
            residual connections, and flexible pooling strategies.
            """

            def __init__(self,
                         node_features: int,
                         hidden_dim: int = 64,
                         output_dim: int = 1,
                         num_conv_layers: int = 3,
                         num_post_layers: int = 1,
                         dropout_rate: float = 0.0,
                         pool_method: str = 'mean',
                         use_batch_norm: bool = True,
                         **kwargs):
                super(GCN, self).__init__(**kwargs)

                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                self.pool_method = pool_method
                self.use_batch_norm = use_batch_norm

                # Initial node embedding
                self.node_embedding = Sequential([
                    layers.Dense(hidden_dim),
                    layers.ReLU()
                ])

                # Graph convolution layers
                self.conv_layers = []
                self.batch_norms = []

                for _ in range(num_conv_layers):
                    self.conv_layers.append(
                        GCNConvolution(hidden_dim, improved=True)
                    )
                    if use_batch_norm:
                        self.batch_norms.append(layers.BatchNormalization())

                # Post-processing layers
                self.post_layers = Sequential([
                    Sequential([
                        layers.Dense(hidden_dim, activation='relu'),
                        layers.Dropout(dropout_rate)
                    ]) for _ in range(num_post_layers)
                ])

                # Output layer
                self.output_layer = layers.Dense(output_dim)

                # Dropout
                self.dropout = layers.Dropout(dropout_rate)

            def pool_nodes(self, x: tf.Tensor, batch_idx: tf.Tensor) -> tf.Tensor:
                """Pool node features to graph-level representations.

                Args:
                    x: Node features [num_nodes, channels]
                    batch_idx: Batch assignments [num_nodes]

                Returns:
                    Graph-level features [num_graphs, channels]
                """
                num_graphs = tf.reduce_max(batch_idx) + 1

                if self.pool_method == 'mean':
                    # Compute mean of node features per graph
                    return tf.scatter_nd(
                        tf.expand_dims(batch_idx, 1),
                        x,
                        shape=[num_graphs, self.hidden_dim]
                    ) / tf.scatter_nd(
                        tf.expand_dims(batch_idx, 1),
                        tf.ones_like(batch_idx, dtype=tf.float32),
                        shape=[num_graphs]
                    )[:, None]

                elif self.pool_method == 'sum':
                    return tf.scatter_nd(
                        tf.expand_dims(batch_idx, 1),
                        x,
                        shape=[num_graphs, self.hidden_dim]
                    )

                elif self.pool_method == 'max':
                    return tf.scatter_nd(
                        tf.expand_dims(batch_idx, 1),
                        x,
                        shape=[num_graphs, self.hidden_dim],
                        reduction='max'
                    )

                raise ValueError(f"Unknown pooling method: {self.pool_method}")

            @tf.function
            def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
                     training: bool = False) -> tf.Tensor:
                """Forward pass of the model.

                Args:
                    inputs: Tuple of (x, edge_index, edge_weight, batch_idx)
                    training: Whether in training mode

                Returns:
                    Predicted properties
                """
                x, edge_index, edge_weight, batch_idx = inputs

                # Initial embedding
                x = self.node_embedding(x)

                # Graph convolution layers
                for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
                    x_new = conv((x, edge_index, edge_weight))
                    if self.use_batch_norm:
                        x_new = bn(x_new, training=training)
                    x_new = tf.nn.relu(x_new)
                    if training:
                        x_new = self.dropout(x_new)

                    # Residual connection if dimensions match
                    if x.shape == x_new.shape:
                        x = x + x_new
                    else:
                        x = x_new

                # Pool nodes to graph representation
                x = self.pool_nodes(x, batch_idx)

                # Post-processing
                x = self.post_layers(x, training=training)

                # Output projection
                out = self.output_layer(x)

                return out if self.output_dim > 1 else tf.squeeze(out, -1)

            def train_step(self, data):
                """Custom training step with gradient clipping.

                Args:
                    data: Tuple of (inputs, targets)

                Returns:
                    Dict of metrics
                """
                x, y = data

                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    loss = self.compiled_loss(y, y_pred)

                # Compute and apply gradients with clipping
                gradients = tape.gradient(loss, self.trainable_variables)
                gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                # Update metrics
                self.compiled_metrics.update_state(y, y_pred)
                metrics = {m.name: m.result() for m in self.metrics}
                metrics['loss'] = loss

                return metrics