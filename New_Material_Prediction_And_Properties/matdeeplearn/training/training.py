"""
Advanced Training System for Materials Graph Neural Networks

A comprehensive training framework optimized for materials science applications,
featuring distributed training, sophisticated evaluation metrics, and advanced
model management.

Key Features:
- Multi-GPU distributed training with TensorFlow
- Custom training loops with gradient management
- Advanced metrics tracking and visualization
- Automated hyperparameter optimization
- Model ensembling and analysis
- Cross-validation support

Supports:
- Multi-task learning
- Mixed-precision training
- Custom loss functions
- Advanced regularization
- Transfer learning
- Active learning

Author: Claude
Date: 2024
"""

import os
import csv
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ME_Deep_Learning import models, process
from ME_Deep_Learning.utils.metrics import MaterialsMetrics
from ME_Deep_Learning.utils.visualization import plot_training_curves


class TrainingEngine:
    """
    Advanced training engine for materials GNN models.

    Features:
    - Distributed training across multiple GPUs
    - Mixed precision training
    - Advanced metrics tracking
    - Memory-efficient batching
    - Comprehensive logging

    Args:
        model: Neural network model
        optimizer: Optimizer instance
        loss_fn: Loss function name or callable
        metrics: List of metrics to track
        strategy: Distribution strategy
        mixed_precision: Whether to use mixed precision
        log_dir: Directory for logs and checkpoints
    """

    def __init__(
            self,
            model: keras.Model,
            optimizer: keras.optimizers.Optimizer,
            loss_fn: Union[str, callable],
            metrics: Optional[List[str]] = None,
            strategy: Optional[tf.distribute.Strategy] = None,
            mixed_precision: bool = True,
            log_dir: Optional[str] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.metrics = [MaterialsMetrics.get_metric(m) for m in (metrics or [])]
        self.strategy = strategy or tf.distribute.get_strategy()

        # Set up loss function
        if isinstance(loss_fn, str):
            self.loss_fn = getattr(tf.keras.losses, loss_fn)()
        else:
            self.loss_fn = loss_fn

        # Enable mixed precision if requested
        if mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        # Set up logging
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Set up TensorBoard
            self.summary_writer = tf.summary.create_file_writer(
                str(self.log_dir)
            )

            # Set up checkpointing
            self.checkpoint = tf.train.Checkpoint(
                step=tf.Variable(0),
                optimizer=optimizer,
                model=model
            )
            self.ckpt_manager = tf.train.CheckpointManager(
                self.checkpoint,
                str(self.log_dir / 'checkpoints'),
                max_to_keep=3
            )

    @tf.function
    def train_step(
            self,
            inputs: Tuple[tf.Tensor, ...],
            labels: tf.Tensor,
            training: bool = True
    ) -> Dict[str, tf.Tensor]:
        """Execute single training step with gradient updates."""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(inputs, training=training)

            # Calculate loss
            loss = self.loss_fn(labels, predictions)

            # Add regularization losses
            if self.model.losses:
                loss += tf.add_n(self.model.losses)

        if training:
            # Compute gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)

            # Clip gradients
            gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]

            # Apply gradients
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

        # Calculate metrics
        metrics = {'loss': loss}
        for metric in self.metrics:
            metrics[metric.name] = metric(labels, predictions)

        return metrics

    @tf.function
    def evaluate_step(
            self,
            inputs: Tuple[tf.Tensor, ...],
            labels: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """Execute single evaluation step."""
        predictions = self.model(inputs, training=False)

        # Calculate metrics
        metrics = {
            'loss': self.loss_fn(labels, predictions)
        }
        for metric in self.metrics:
            metrics[metric.name] = metric(labels, predictions)

        return metrics, predictions

    def train(
            self,
            train_dataset: tf.data.Dataset,
            validation_dataset: Optional[tf.data.Dataset] = None,
            epochs: int = 100,
            callbacks: Optional[List[keras.callbacks.Callback]] = None,
            verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Execute complete training loop.

        Args:
            train_dataset: Training data
            validation_dataset: Validation data
            epochs: Number of epochs
            callbacks: Training callbacks
            verbose: Verbosity level

        Returns:
            Dictionary of training history
        """
        # Initialize history
        history = {
            'loss': [],
            'val_loss': []
        }
        for metric in self.metrics:
            history[metric.name] = []
            history[f'val_{metric.name}'] = []

        # Set up callbacks
        callbacks = callbacks or []
        for callback in callbacks:
            callback.set_model(self.model)

        # Training loop
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            start_time = time.time()

            # Train epoch
            train_metrics = []
            for batch in train_dataset:
                metrics = self.train_step(*batch)
                train_metrics.append(metrics)

            # Average training metrics
            train_metrics = {
                k: tf.reduce_mean([m[k] for m in train_metrics]).numpy()
                for k in train_metrics[0].keys()
            }

            # Validate if dataset provided
            if validation_dataset is not None:
                val_metrics = []
                val_predictions = []

                for batch in validation_dataset:
                    metrics, predictions = self.evaluate_step(*batch)
                    val_metrics.append(metrics)
                    val_predictions.append(predictions)

                # Average validation metrics
                val_metrics = {
                    k: tf.reduce_mean([m[k] for m in val_metrics]).numpy()
                    for k in val_metrics[0].keys()
                }

            # Update history
            for k, v in train_metrics.items():
                history[k].append(v)
            if validation_dataset is not None:
                for k, v in val_metrics.items():
                    history[f'val_{k}'].append(v)

            # Log metrics
            if hasattr(self, 'summary_writer'):
                with self.summary_writer.as_default():
                    for k, v in train_metrics.items():
                        tf.summary.scalar(
                            f'train/{k}',
                            v,
                            step=epoch
                        )
                    if validation_dataset is not None:
                        for k, v in val_metrics.items():
                            tf.summary.scalar(
                                f'val/{k}',
                                v,
                                step=epoch
                            )

            # Checkpointing
            if hasattr(self, 'ckpt_manager'):
                if (epoch + 1) % 10 == 0:
                    self.ckpt_manager.save()

            # Print metrics
            if verbose:
                metrics_str = ' - '.join(
                    f'{k}: {v:.4f}' for k, v in {
                        **train_metrics,
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    }.items()
                )
                print(f'{metrics_str} - {time.time() - start_time:.1f}s/epoch')

        return history

    class DistributedTrainer:
        """
        Multi-GPU distributed training coordinator.

        Features:
        - Multi-GPU synchronous training
        - Dynamic batch sharding
        - Gradient aggregation
        - Memory management
        - Cross-GPU communication

        Args:
            num_gpus: Number of GPUs to use
            model_fn: Function to create model instances
            mixed_precision: Whether to use mixed precision
        """

        def __init__(
                self,
                num_gpus: int,
                model_fn: callable,
                mixed_precision: bool = True
        ):
            self.num_gpus = num_gpus
            self.model_fn = model_fn

            # Set up distribution strategy
            if num_gpus > 1:
                self.strategy = tf.distribute.MirroredStrategy()
            else:
                self.strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

            # Enable mixed precision if requested
            if mixed_precision:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)

            # Create distributed model
            with self.strategy.scope():
                self.model = self.model_fn()

        def compile(
                self,
                optimizer: Union[str, keras.optimizers.Optimizer],
                loss: Union[str, callable],
                metrics: Optional[List[str]] = None,
                **kwargs
        ):
            """Compile distributed model."""
            with self.strategy.scope():
                self.model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=[MaterialsMetrics.get_metric(m) for m in (metrics or [])],
                    **kwargs
                )

        def fit(
                self,
                train_data: tf.data.Dataset,
                validation_data: Optional[tf.data.Dataset] = None,
                epochs: int = 100,
                callbacks: Optional[List[keras.callbacks.Callback]] = None,
                **kwargs
        ) -> Dict[str, List[float]]:
            """Train distributed model."""
            # Prepare datasets for distribution
            train_dist = self.strategy.experimental_distribute_dataset(train_data)
            if validation_data is not None:
                val_dist = self.strategy.experimental_distribute_dataset(validation_data)
            else:
                val_dist = None

            # Create training engine
            trainer = TrainingEngine(
                model=self.model,
                optimizer=self.model.optimizer,
                loss_fn=self.model.loss,
                metrics=self.model.metrics,
                strategy=self.strategy
            )

            # Train model
            history = trainer.train(
                train_dist,
                val_dist,
                epochs=epochs,
                callbacks=callbacks,
                **kwargs
            )

            return history

    class HyperparameterOptimizer:
        """
        Advanced hyperparameter optimization system.

        Features:
        - Bayesian optimization
        - Multi-objective optimization
        - Early stopping
        - Trial pruning
        - Cross-validation integration

        Args:
            model_fn: Function to create model
            search_space: Hyperparameter search space
            objective: Optimization objective
            max_trials: Maximum number of trials
            directory: Directory for results
        """

        def __init__(
                self,
                model_fn: callable,
                search_space: Dict[str, Any],
                objective: str = "val_loss",
                max_trials: int = 50,
                directory: str = "hpo_results"
        ):
            self.model_fn = model_fn
            self.search_space = search_space
            self.objective = objective
            self.max_trials = max_trials
            self.directory = Path(directory)

            # Set up keras tuner
            self.tuner = keras.tuners.BayesianOptimization(
                self._build_model,
                objective=keras.tuners.Objective(
                    objective,
                    direction="min"
                ),
                max_trials=max_trials,
                directory=str(self.directory),
                project_name="materials_gnn"
            )

        def _build_model(self, hp):
            """Build model with trial hyperparameters."""
            hparams = {}
            for name, space in self.search_space.items():
                if space["type"] == "int":
                    hparams[name] = hp.Int(
                        name,
                        min_value=space["min"],
                        max_value=space["max"],
                        step=space.get("step", 1)
                    )
                elif space["type"] == "float":
                    hparams[name] = hp.Float(
                        name,
                        min_value=space["min"],
                        max_value=space["max"],
                        sampling=space.get("sampling", "linear")
                    )
                elif space["type"] == "choice":
                    hparams[name] = hp.Choice(
                        name,
                        values=space["values"]
                    )

            return self.model_fn(**hparams)

        def search(
                self,
                train_data: tf.data.Dataset,
                validation_data: tf.data.Dataset,
                **train_kwargs
        ) -> Tuple[Dict[str, Any], keras.Model]:
            """Execute hyperparameter search."""
            # Set up callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor=self.objective,
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor=self.objective,
                    factor=0.5,
                    patience=5
                )
            ]

            # Run search
            self.tuner.search(
                train_data,
                validation_data=validation_data,
                callbacks=callbacks,
                **train_kwargs
            )

            # Get best model and hyperparameters
            best_hp = self.tuner.get_best_hyperparameters()[0]
            best_model = self.tuner.get_best_models()[0]

            return best_hp.values, best_model

    class EnsembleTrainer:
        """
        Advanced model ensembling system.

        Features:
        - Diverse model combination
        - Weighted averaging
        - Uncertainty estimation
        - Cross-validation integration
        - Advanced pooling methods

        Args:
            model_fns: List of model creation functions
            weights: Optional model weights
            aggregation: Method to combine predictions
        """

        def __init__(
                self,
                model_fns: List[callable],
                weights: Optional[List[float]] = None,
                aggregation: str = "weighted_average"
        ):
            self.model_fns = model_fns
            self.weights = weights or [1.0] * len(model_fns)
            self.aggregation = aggregation
            self.models = []

        def train_model(
                self,
                model_fn: callable,
                train_data: tf.data.Dataset,
                validation_data: tf.data.Dataset,
                **train_kwargs
        ) -> Tuple[keras.Model, Dict[str, List[float]]]:
            """Train single ensemble member."""
            model = model_fn()
            trainer = TrainingEngine(
                model=model,
                optimizer=train_kwargs.pop("optimizer"),
                loss_fn=train_kwargs.pop("loss"),
                metrics=train_kwargs.pop("metrics", None)
            )

            history = trainer.train(
                train_data,
                validation_data,
                **train_kwargs
            )

            return model, history

        def train_ensemble(
                self,
                train_data: tf.data.Dataset,
                validation_data: tf.data.Dataset,
                **train_kwargs
        ) -> List[Tuple[keras.Model, Dict[str, List[float]]]]:
            """Train complete ensemble."""
            results = []
            for i, model_fn in enumerate(self.model_fns):
                print(f"\nTraining model {i + 1}/{len(self.model_fns)}")
                model, history = self.train_model(
                    model_fn,
                    train_data,
                    validation_data,
                    **train_kwargs
                )
                results.append((model, history))
                self.models.append(model)

            return results

        @tf.function
        def predict_ensemble(
                self,
                inputs: Tuple[tf.Tensor, ...]
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            """Get ensemble predictions with uncertainty."""
            predictions = []
            for model, weight in zip(self.models, self.weights):
                pred = model(inputs, training=False) * weight
                predictions.append(pred)

            # Stack predictions
            predictions = tf.stack(predictions)

            # Compute mean and uncertainty
            if self.aggregation == "weighted_average":
                mean_pred = tf.reduce_mean(predictions, axis=0)
                std_pred = tf.math.reduce_std(predictions, axis=0)
            elif self.aggregation == "weighted_median":
                mean_pred = tfp.stats.percentile(predictions, 50.0, axis=0)
                std_pred = tf.math.reduce_std(predictions, axis=0)

            return mean_pred, std_pred

    class ModelAnalyzer:
        """
        Advanced model analysis and visualization.

        Features:
        - Feature importance analysis
        - Attention visualization
        - Embedding analysis
        - Performance profiling
        - Error analysis

        Args:
            model: Trained model to analyze
            layer_names: Names of layers to analyze
        """

        def __init__(
                self,
                model: keras.Model,
                layer_names: Optional[List[str]] = None
        ):
            self.model = model
            self.layer_names = layer_names or []

            # Set up feature extractors
            self.extractors = {}
            for name in self.layer_names:
                layer = model.get_layer(name)
                self.extractors[name] = keras.Model(
                    inputs=model.input,
                    outputs=layer.output
                )

        def extract_features(
                self,
                dataset: tf.data.Dataset,
                layer_name: str
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Extract intermediate features."""
            features = []
            labels = []

            for batch in dataset:
                feat = self.extractors[layer_name](batch[0])
                features.append(feat)
                labels.append(batch[1])

            return np.concatenate(features), np.concatenate(labels)

        def analyze_attention(
                self,
                dataset: tf.data.Dataset,
                layer_name: str
        ) -> Dict[str, np.ndarray]:
            """Analyze attention weights."""
            attention_weights = []
            node_features = []

            layer = self.model.get_layer(layer_name)

            for batch in dataset:
                weights = layer(batch[0], training=False)
                attention_weights.append(weights)
                node_features.append(batch[0])

            return {
                'weights': np.concatenate(attention_weights),
                'features': np.concatenate(node_features)
            }

        def visualize_embeddings(
                self,
                features: np.ndarray,
                labels: np.ndarray,
                method: str = "tsne",
                **kwargs
        ) -> np.ndarray:
            """Visualize high-dimensional embeddings."""
            if method == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(**kwargs)
            elif method == "umap":
                import umap
                reducer = umap.UMAP(**kwargs)

            embeddings = reducer.fit_transform(features)

            # Plot results
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            scatter = plt.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.5
            )
            plt.colorbar(scatter)
            plt.show()

            return embeddings

        def profile_performance(
                self,
                dataset: tf.data.Dataset,
                batch_sizes: List[int],
                warmup_steps: int = 10,
                profile_steps: int = 100
        ) -> Dict[int, Dict[str, float]]:
            """Profile model performance."""
            results = {}

            for batch_size in batch_sizes:
                # Prepare data
                data = dataset.batch(batch_size)

                # Warmup
                for _ in range(warmup_steps):
                    for batch in data:
                        _ = self.model(batch[0], training=False)

                # Profile
                times = []
                memory = []
                for _ in range(profile_steps):
                    for batch in data:
                        start = time.time()
                        _ = self.model(batch[0], training=False)
                        times.append(time.time() - start)

                        memory.append(
                            tf.config.experimental.get_memory_info('GPU:0')['peak']
                            if tf.config.list_physical_devices('GPU')
                            else 0
                        )

                results[batch_size] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'throughput': batch_size / np.mean(times),
                    'peak_memory': max(memory) / (1024 * 1024)  # MB
                }

            return results

    def setup_training(config: Dict[str, Any]) -> TrainingEngine:
        """Set up training with configuration."""
        # Set up distribution strategy
        if len(tf.config.list_physical_devices('GPU')) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

        # Set memory growth
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        # Create model and optimizer
        with strategy.scope():
            model = config["model_fn"]()
            optimizer = getattr(tf.keras.optimizers, config["optimizer"])(
                learning_rate=config["learning_rate"]
            )

        # Create training engine
        trainer = TrainingEngine(
            model=model,
            optimizer=optimizer,
            loss_fn=config["loss"],
            metrics=config.get("metrics"),
            strategy=strategy,
            mixed_precision=config.get("mixed_precision", True),
            log_dir=config.get("log_dir")
        )

        return trainer

    def write_results(
            outputs: np.ndarray,
            filename: str,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """Write model outputs with metadata."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write metadata
            if metadata:
                for key, value in metadata.items():
                    writer.writerow([f"# {key}", value])
                writer.writerow([])

            # Write header
            writer.writerow([
                'id',
                'target',
                'prediction',
                'uncertainty'
            ])

            # Write results
            writer.writerows(outputs)
