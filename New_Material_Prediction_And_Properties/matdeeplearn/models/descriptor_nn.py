import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from typing import List, Optional


class DescriptorNetwork(Model):
    """Base class for descriptor-based neural networks.

    This class provides common functionality for networks that operate on
    pre-computed molecular/crystal descriptors.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 output_dim: int = 1,
                 dropout_rate: float = 0.0,
                 activation: str = 'relu',
                 **kwargs):
        super(DescriptorNetwork, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Build network layers
        layers_list = []

        # First layer
        layers_list.extend([
            layers.Dense(hidden_dim),
            layers.Activation(activation),
            layers.Dropout(dropout_rate)
        ])

        # Hidden layers
        for _ in range(num_layers - 1):
            layers_list.extend([
                layers.Dense(hidden_dim),
                layers.Activation(activation),
                layers.Dropout(dropout_rate)
            ])

        # Output layer
        layers_list.append(layers.Dense(output_dim))

        self.network = Sequential(layers_list)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Input descriptors [batch_size, input_dim]
            training: Whether in training mode

        Returns:
            Predicted properties [batch_size, output_dim]
        """
        out = self.network(inputs, training=training)
        return out if self.output_dim > 1 else tf.squeeze(out, -1)


class SOAP(DescriptorNetwork):
    """Neural network for Smooth Overlap of Atomic Positions (SOAP) descriptors.

    This model processes SOAP descriptors to predict material properties.
    SOAP descriptors capture local atomic environments through expansion
    in terms of atomic density overlap.
    """

    def __init__(self,
                 soap_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 output_dim: int = 1,
                 **kwargs):
        super(SOAP, self).__init__(
            input_dim=soap_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            name='SOAP_Network',
            **kwargs
        )

    def call(self, data, training: bool = False) -> tf.Tensor:
        """Process SOAP descriptors.

        Args:
            data: Object containing SOAP descriptors in extra_features_SOAP
            training: Whether in training mode

        Returns:
            Predicted properties
        """
        return super().call(data.extra_features_SOAP, training=training)


class SM(DescriptorNetwork):
    """Neural network for Sine Matrix descriptors.

    This model processes Sine Matrix descriptors to predict material properties.
    The Sine Matrix captures periodicity and structural information of crystals
    through trigonometric functions of atomic positions.
    """

    def __init__(self,
                 sm_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 output_dim: int = 1,
                 **kwargs):
        super(SM, self).__init__(
            input_dim=sm_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            name='SM_Network',
            **kwargs
        )

    def call(self, data, training: bool = False) -> tf.Tensor:
        """Process Sine Matrix descriptors.

        Args:
            data: Object containing Sine Matrix descriptors in extra_features_SM
            training: Whether in training mode

        Returns:
            Predicted properties
        """
        return super().call(data.extra_features_SM, training=training)