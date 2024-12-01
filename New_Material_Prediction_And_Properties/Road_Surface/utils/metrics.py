import tensorflow as tf
import numpy as np
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive metrics for model evaluation."""
    metrics = {
        'mae': tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy(),
        'rmse': np.sqrt(tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy()),
        'r2': r2_score(y_true, y_pred),
        'max_error': np.max(np.abs(y_true - y_pred)),
        'structural_safety_factor': calculate_safety_factor(y_true, y_pred)
    }
    return metrics


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / (ss_tot + 1e-7))


def calculate_safety_factor(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate structural safety factor for municipal approval."""
    relative_error = np.abs((y_true - y_pred) / y_true)
    return 1.0 / (1.0 + np.mean(relative_error))


def evaluate_model(model: tf.keras.Model,
                   test_data: Tuple[np.ndarray, np.ndarray],
                   uncertainty: bool = True) -> Dict[str, float]:
    """Evaluate model with uncertainty estimation."""
    x_test, y_test = test_data

    if uncertainty:
        predictions = []
        for _ in range(10):  # Monte Carlo sampling
            pred = model(x_test, training=True)
            predictions.append(pred)

        y_pred = tf.reduce_mean(predictions, axis=0)
        uncertainty = tf.math.reduce_std(predictions, axis=0)
    else:
        y_pred = model(x_test, training=False)
        uncertainty = None

    metrics = calculate_metrics(y_test, y_pred)
    if uncertainty is not None:
        metrics['prediction_uncertainty'] = tf.reduce_mean(uncertainty).numpy()

    return metrics