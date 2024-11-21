"""
Comprehensive utilities for materials GNN analysis and optimization.

Provides tools for:
- Model analysis and visualization
- Performance profiling
- Memory usage tracking
- Training diagnostics
- Custom metrics
- Debugging tools

Author: Claude
Date: 2024
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
import logging
import psutil
import os
from pathlib import Path


class MemoryTracker:
    """Tracks memory usage during model operations."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                filename=f"{log_dir}/memory_usage.log",
                level=logging.INFO
            )

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        stats = {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
        }

        if tf.config.list_physical_devices('GPU'):
            for device in tf.config.list_physical_devices('GPU'):
                with tf.device(device.name):
                    stats[f"gpu_{device.name}_mb"] = (
                            tf.config.experimental.get_memory_info(device.name)['current']
                            / (1024 * 1024)
                    )

        return stats

    def log_memory(self, tag: str):
        """Log current memory usage with a tag."""
        stats = self.get_memory_usage()
        message = f"{tag} - " + " | ".join(
            f"{k}: {v:.2f}MB" for k, v in stats.items()
        )
        if self.log_dir:
            logging.info(message)
        else:
            print(message)


def profile_model(
        model: tf.keras.Model,
        input_data: Union[tf.Tensor, List[tf.Tensor]],
        batch_sizes: List[int] = [32, 64, 128, 256],
        warmup_runs: int = 10,
        profile_runs: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive model profiling across batch sizes.

    Args:
        model: Model to profile
        input_data: Sample input data
        batch_sizes: Batch sizes to test
        warmup_runs: Number of warmup iterations
        profile_runs: Number of profiling iterations

    Returns:
        Dictionary of profiling metrics per batch size
    """
    results = {}
    memory_tracker = MemoryTracker()

    for batch_size in batch_sizes:
        # Prepare batch data
        if isinstance(input_data, list):
            batch = [tf.repeat(x[:1], batch_size, axis=0) for x in input_data]
        else:
            batch = tf.repeat(input_data[:1], batch_size, axis=0)

        # Warmup
        for _ in range(warmup_runs):
            _ = model(batch, training=False)

        # Profile
        times = []
        memory_peaks = []

        for _ in range(profile_runs):
            memory_tracker.log_memory(f"Before run (batch_size={batch_size})")

            start = time.perf_counter()
            _ = model(batch, training=False)
            end = time.perf_counter()

            times.append(end - start)
            memory_stats = memory_tracker.get_memory_usage()
            memory_peaks.append(max(memory_stats.values()))

            memory_tracker.log_memory(f"After run (batch_size={batch_size})")

        results[batch_size] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "throughput": batch_size / np.mean(times),
            "peak_memory_mb": max(memory_peaks),
            "memory_per_sample_mb": max(memory_peaks) / batch_size
        }

    return results


def analyze_gradients(
        model: tf.keras.Model,
        data: Tuple[tf.Tensor, tf.Tensor],
        aggregation: str = "mean"
) -> Dict[str, tf.Tensor]:
    """
    Analyze gradient statistics during training.

    Args:
        model: Model to analyze
        data: (inputs, targets) tuple
        aggregation: How to aggregate gradient stats ("mean" or "norm")

    Returns:
        Dictionary of gradient statistics per layer
    """
    x, y = data

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = model.compiled_loss(y, y_pred)

    # Get gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Compute statistics
    stats = {}
    for var, grad in zip(model.trainable_variables, grads):
        if grad is not None:
            if aggregation == "mean":
                stats[var.name] = tf.reduce_mean(tf.abs(grad))
            else:
                stats[var.name] = tf.norm(grad)

    return stats


def visualize_attention(
        model: tf.keras.Model,
        input_data: Union[tf.Tensor, List[tf.Tensor]],
        layer_name: str
) -> tf.Tensor:
    """
    Visualize attention weights from a specific layer.

    Args:
        model: Model containing attention layers
        input_data: Input data to process
        layer_name: Name of attention layer to visualize

    Returns:
        Attention weights tensor
    """
    # Get attention layer
    attention_layer = None
    for layer in model.layers:
        if layer.name == layer_name:
            attention_layer = layer
            break

    if attention_layer is None:
        raise ValueError(f"No layer found with name {layer_name}")

    # Create intermediate model
    intermediate_model = tf.keras.Model(
        inputs=model.input,
        outputs=attention_layer.output
    )

    # Get attention weights
    attention_weights = intermediate_model(input_data)
    return attention_weights


class CustomMetrics:
    """Collection of custom metrics for materials property prediction."""

    @staticmethod
    def mean_absolute_relative_error(
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Compute mean absolute relative error."""
        return tf.reduce_mean(
            tf.abs((y_true - y_pred) / (tf.abs(y_true) + 1e-7))
        )

    @staticmethod
    def root_mean_squared_log_error(
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Compute root mean squared logarithmic error."""
        return tf.sqrt(
            tf.reduce_mean(
                tf.square(tf.math.log1p(y_true) - tf.math.log1p(y_pred))
            )
        )

    @staticmethod
    def weighted_mae(
            weights: tf.Tensor
    ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Create weighted mean absolute error metric."""

        def weighted_mae_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return tf.reduce_mean(weights * tf.abs(y_true - y_pred))

        return weighted_mae_fn


class TrainingAnalyzer:
    """Analyzes and tracks training progress."""

    def __init__(
            self,
            model: tf.keras.Model,
            log_dir: str,
            metrics: List[str]
    ):
        self.model = model
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}

    def on_epoch_end(
            self,
            epoch: int,
            logs: Optional[Dict[str, float]] = None
    ):
        """Record metrics at epoch end."""
        if logs:
            for metric in self.metrics:
                if metric in logs:
                    self.history[metric].append(logs[metric])

        # Save metrics
        np.save(
            f"{self.log_dir}/training_history.npy",
            self.history
        )

    def plot_metrics(self):
        """Plot training metrics over time."""
        import matplotlib.pyplot as plt

        for metric in self.metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history[metric])
            plt.title(f"{metric} Over Time")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.savefig(f"{self.log_dir}/{metric}_plot.png")
            plt.close()

    def analyze_convergence(
            self,
            window_size: int = 10,
            threshold: float = 0.01
    ) -> Dict[str, bool]:
        """
        Analyze convergence of each metric.

        Args:
            window_size: Size of window for moving average
            threshold: Convergence threshold

        Returns:
            Dictionary indicating convergence status of each metric
        """
        convergence = {}

        for metric in self.metrics:
            values = self.history[metric]
            if len(values) < window_size:
                convergence[metric] = False
                continue

            # Compute moving average
            moving_avg = np.convolve(
                values,
                np.ones(window_size) / window_size,
                mode='valid'
            )

            # Check if change is below threshold
            changes = np.abs(np.diff(moving_avg))
            convergence[metric] = np.all(changes[-5:] < threshold)

        return convergence


def debug_nan_gradients(
        model: tf.keras.Model,
        data: Tuple[tf.Tensor, tf.Tensor]
) -> Dict[str, bool]:
    """
    Debug NaN gradients during training.

    Args:
        model: Model to debug
        data: (inputs, targets) tuple

    Returns:
        Dictionary indicating which layers have NaN gradients
    """
    x, y = data

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = model.compiled_loss(y, y_pred)

    # Get gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Check for NaNs
    nan_layers = {}
    for var, grad in zip(model.trainable_variables, grads):
        if grad is not None:
            nan_layers[var.name] = tf.reduce_any(tf.math.is_nan(grad))

    return nan_layers