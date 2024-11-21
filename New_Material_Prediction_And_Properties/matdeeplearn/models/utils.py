import tensorflow as tf
from tensorflow.keras import Model
from typing import Dict, List, Optional
import numpy as np


def model_summary(model: Model,
                  print_fn: Optional[callable] = None,
                  expand_nested: bool = False) -> Dict[str, int]:
    """Enhanced model summary with detailed parameter analysis.

    This function provides a comprehensive analysis of model architecture
    and parameters with additional insights for materials science models.

    Args:
        model: TensorFlow/Keras model
        print_fn: Optional custom print function
        expand_nested: Whether to expand nested models

    Returns:
        Dict containing parameter statistics
    """
    if print_fn is None:
        print_fn = print

    def format_size(size: int) -> str:
        """Format parameter size with appropriate units."""
        if size < 1000:
            return str(size)
        elif size < 1000000:
            return f"{size / 1000:.1f}K"
        else:
            return f"{size / 1000000:.1f}M"

    print_fn("=" * 80)
    print_fn(f"Model: {model.__class__.__name__}")
    print_fn("=" * 80)

    # Collect all layers including nested ones if specified
    layers = []

    def collect_layers(layer):
        layers.append(layer)
        if expand_nested and hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                collect_layers(sublayer)

    collect_layers(model)

    # Format header
    header = "{:<50} {:<20} {:<15}".format(
        "Layer (type)", "Output Shape", "Param #"
    )
    print_fn("=" * 85)
    print_fn(header)
    print_fn("=" * 85)

    # Track parameter statistics
    total_params = 0
    trainable_params = 0
    layer_stats = {}

    # Print layer information
    for layer in layers:
        name = layer.name
        layer_type = layer.__class__.__name__

        # Get output shape
        if hasattr(layer, 'output_shape'):
            output_shape = str(layer.output_shape)
        else:
            output_shape = 'multiple' if hasattr(layer, 'output_shapes') else 'unknown'

        # Count parameters
        params = layer.count_params()
        trainable_params_layer = sum(
            tf.size(w).numpy() for w in layer.trainable_weights
        )
        non_trainable_params = params - trainable_params_layer

        # Update totals
        total_params += params
        trainable_params += trainable_params_layer

        # Store layer statistics
        layer_stats[name] = {
            'type': layer_type,
            'output_shape': output_shape,
            'params': params,
            'trainable_params': trainable_params_layer,
            'non_trainable_params': non_trainable_params
        }

        # Print layer info
        layer_line = "{:<50} {:<20} {:<15}".format(
            f"{name} ({layer_type})",
            output_shape,
            format_size(params)
        )
        print_fn(layer_line)

    print_fn("=" * 85)

    # Print summary statistics
    print_fn(f"Total parameters: {format_size(total_params)}")
    print_fn(f"Trainable parameters: {format_size(trainable_params)}")
    print_fn(f"Non-trainable parameters: {format_size(total_params - trainable_params)}")

    # Memory usage estimation
    param_bytes = total_params * 4  # Assuming float32
    print_fn(f"Estimated model size: {format_size(param_bytes)} bytes")

    # Additional insights for materials science models
    if hasattr(model, 'cutoff'):
        print_fn(f"Interaction cutoff: {model.cutoff:.2f} Å")
    if hasattr(model, 'hidden_dim'):
        print_fn(f"Feature dimension: {model.hidden_dim}")

    # Return detailed statistics
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'param_bytes': param_bytes,
        'layer_stats': layer_stats
    }


def analyze_model_capacity(model: Model) -> Dict[str, float]:
    """Analyze model capacity and complexity metrics.

    This function computes various metrics to assess model capacity
    and potential expressiveness for materials property prediction.

    Args:
        model: TensorFlow/Keras model

    Returns:
        Dict containing capacity metrics
    """
    # Count parameters by type
    conv_params = sum(
        layer.count_params()
        for layer in model.layers
        if 'conv' in layer.__class__.__name__.lower()
    )

    dense_params = sum(
        layer.count_params()
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    )

    # Compute complexity metrics
    total_params = model.count_params()
    depth = len([
        layer for layer in model.layers
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D))
    ])

    # Estimate theoretical capacity
    log_capacity = np.log10(total_params * depth)

    # Compute parameter efficiency
    conv_ratio = conv_params / total_params if total_params > 0 else 0
    dense_ratio = dense_params / total_params if total_params > 0 else 0

    return {
        'total_params': total_params,
        'conv_params': conv_params,
        'dense_params': dense_params,
        'depth': depth,
        'log_capacity': log_capacity,
        'conv_ratio': conv_ratio,
        'dense_ratio': dense_ratio
    }