import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional
import tensorflow as tf


def plot_history(history: Dict[str, list],
                 output_path: Optional[str] = None):
    """Plot training history with confidence intervals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.fill_between(epochs,
                     np.array(history['loss']) - np.array(history.get('loss_std', 0)),
                     np.array(history['loss']) + np.array(history.get('loss_std', 0)),
                     alpha=0.2)
    ax1.set_title('Model Loss')
    ax1.legend()

    # Metrics plot
    for metric in history:
        if metric not in ['loss', 'val_loss']:
            ax2.plot(epochs, history[metric], label=metric)
    ax2.set_title('Model Metrics')
    ax2.legend()

    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     uncertainty: Optional[np.ndarray] = None,
                     output_path: Optional[str] = None):
    """Plot predictions vs actual values with uncertainty."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5)
    ax1.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')

    # Error distribution
    errors = y_pred - y_true
    sns.histplot(errors, kde=True, ax=ax2)
    ax2.set_title('Prediction Error Distribution')

    if uncertainty is not None:
        ax1.fill_between(y_true,
                         y_pred - 2 * uncertainty,
                         y_pred + 2 * uncertainty,
                         alpha=0.2)

    if output_path:
        plt.savefig(output_path)
    plt.show()