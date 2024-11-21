from .training import (
    # Core classes
    TrainingConfig,
    CrossValidator,
    EnsembleTrainer,

    # Training utilities
    ModelCheckpoint,
    EarlyStopping,

    # Configuration dataclasses
    TrainingConfig,
    OptimizerConfig,

    # Custom metrics
    MaterialPropertyMetrics,
    StructureMetrics,

    # Type definitions
    ModelBuilder,
    DatasetType,
    MetricType
)

import logging
import tensorflow as tf
from typing import Dict, Any

# Setup logging
logger = logging.getLogger("materials_training")
logger.setLevel(logging.INFO)

# Default configurations
DEFAULT_TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 10,
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae', 'rmse'],
    'validation_split': 0.2,
    'test_split': 0.1
}

DEFAULT_ENSEMBLE_CONFIG = {
    'n_models': 5,
    'weights_method': 'performance',
    'use_uncertainty': True,
    'diversity_metric': 'correlation'
}

DEFAULT_CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'stratify': True
}


# Initialize hardware-specific settings
def initialize_training(
        mixed_precision: bool = True,
        memory_growth: bool = True,
        log_device_placement: bool = False
) -> None:
    """Initialize training environment with hardware optimizations.

    Args:
        mixed_precision: Whether to use mixed precision training
        memory_growth: Whether to enable memory growth
        log_device_placement: Whether to log device placement
    """
    # Set up mixed precision
    if mixed_precision and tf.config.list_physical_devices('GPU'):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("Enabled mixed precision training")

    # Configure GPU memory growth
    if memory_growth:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("Enabled GPU memory growth")

    # Set device placement logging
    tf.debugging.set_log_device_placement(log_device_placement)


# Custom metrics for materials science
def register_custom_metrics() -> None:
    """Register custom metrics for materials property prediction."""
    from tensorflow.keras.metrics import Metric

    custom_metrics = {
        'mean_absolute_relative_error': lambda: MaterialPropertyMetrics.MARE(),
        'max_absolute_error': lambda: MaterialPropertyMetrics.MaxAE(),
        'structure_similarity': lambda: StructureMetrics.StructureSimilarity(),
        'composition_error': lambda: StructureMetrics.CompositionError()
    }

    for name, metric_fn in custom_metrics.items():
        tf.keras.utils.get_custom_objects()[name] = metric_fn

    logger.info(f"Registered {len(custom_metrics)} custom metrics")


# Version information
__version__ = '2.0.0'


# Package initialization
def configure_training(config: Dict[str, Any] = None) -> TrainingConfig:
    """Configure training with custom settings.

    Args:
        config: Optional configuration overrides

    Returns:
        TrainingConfig instance
    """
    # Merge with defaults
    full_config = {**DEFAULT_TRAINING_CONFIG, **(config or {})}

    # Create config instance
    training_config = TrainingConfig(**full_config)

    # Initialize environment
    initialize_training()

    # Register custom metrics
    register_custom_metrics()

    return training_config


# Hardware detection and optimization
def get_optimal_batch_size(
        model: tf.keras.Model,
        start_batch_size: int = 32
) -> int:
    """Determine optimal batch size for available hardware.

    Args:
        model: Model instance
        start_batch_size: Initial batch size to try

    Returns:
        Optimal batch size
    """
    if not tf.config.list_physical_devices('GPU'):
        return start_batch_size

    batch_size = start_batch_size
    while True:
        try:
            # Try to create a batch of random data
            dummy_data = tf.random.normal([batch_size, *model.input_shape[1:]])
            model(dummy_data, training=False)
            batch_size *= 2
        except tf.errors.ResourceExhaustedError:
            # If we run out of memory, use the previous batch size
            return batch_size // 2


def get_distribution_strategy() -> tf.distribute.Strategy:
    """Get appropriate distribution strategy for available hardware."""
    if len(tf.config.list_physical_devices('GPU')) > 1:
        return tf.distribute.MirroredStrategy()
    elif tf.config.list_physical_devices('GPU'):
        return tf.distribute.OneDeviceStrategy("/gpu:0")
    else:
        return tf.distribute.OneDeviceStrategy("/cpu:0")


# Error handling
class MaterialsTrainingError(Exception):
    """Base exception class for materials training errors."""
    pass


class ModelConfigurationError(MaterialsTrainingError):
    """Raised when model configuration is invalid."""
    pass


class DatasetError(MaterialsTrainingError):
    """Raised when there are dataset-related issues."""
    pass


class HardwareError(MaterialsTrainingError):
    """Raised when hardware configuration is problematic."""
    pass


# Export all
__all__ = [
    # Core classes
    'TrainingConfig',
    'CrossValidator',
    'EnsembleTrainer',
    'ModelCheckpoint',
    'EarlyStopping',

    # Configuration
    'configure_training',
    'initialize_training',
    'get_optimal_batch_size',
    'get_distribution_strategy',

    # Constants
    'DEFAULT_TRAINING_CONFIG',
    'DEFAULT_ENSEMBLE_CONFIG',
    'DEFAULT_CV_CONFIG',

    # Error classes
    'MaterialsTrainingError',
    'ModelConfigurationError',
    'DatasetError',
    'HardwareError'
]
