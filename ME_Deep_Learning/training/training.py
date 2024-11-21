"""
Core training functionality for materials GNN models.

Implements sophisticated training loops and evaluation methods with support for:
- Distributed training
- Multi-GPU parallelization
- Custom learning schedules
- Advanced regularization
- Comprehensive metrics tracking
- Memory-efficient batching
- Model checkpointing

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
from rich import json
from tensorflow import keras

from ME_Deep_Learning import models
from ME_Deep_Learning.preprocessing import process
from ME_Deep_Learning.utils.metrics import MaterialsMetrics
from ME_Deep_Learning.utils.visualization import ModelVisualizer


class TrainingEngine:
    """
    Core training engine handling model training and evaluation.

    Features:
    - Distributed training support
    - Advanced metrics tracking
    - Memory-efficient batching
    - Comprehensive logging
    - Model checkpointing

    Args:
        strategy: Distribution strategy
        model: Model to train
        optimizer: Optimizer instance
        loss_fn: Loss function
        metrics: List of metrics to track
        checkpoint_dir: Directory for checkpoints
    """

    def __init__(
            self,
            strategy: tf.distribute.Strategy,
            model: keras.Model,
            optimizer: keras.optimizers.Optimizer,
            loss_fn: str,
            metrics: Optional[List[str]] = None,
            checkpoint_dir: Optional[str] = None
    ):
        self.strategy = strategy
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = getattr(tf.keras.losses, loss_fn)()
        self.metrics = [MaterialsMetrics.get_metric(m) for m in (metrics or [])]

        # Set up checkpointing
        if checkpoint_dir:
            self.ckpt = tf.train.Checkpoint(
                step=tf.Variable(0),
                optimizer=optimizer,
                model=model
            )
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt,
                checkpoint_dir,
                max_to_keep=3
            )

    @tf.function
    def train_step(
            self,
            inputs: Tuple[tf.Tensor, ...],
            labels: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Single training step with gradient updates."""
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)

        # Compute gradients and update
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update metrics
        metrics = {'loss': loss}
        for metric in self.metrics:
            metrics[metric.name] = metric(labels, predictions)

        return metrics

    @tf.function
    def evaluate_step(
            self,
            inputs: Tuple[tf.Tensor, ...],
            labels: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Single evaluation step."""
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(labels, predictions)

        metrics = {'loss': loss}
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
        Full training loop with validation.

        Args:
            train_dataset: Training data
            validation_dataset: Validation data
            epochs: Number of epochs
            callbacks: Training callbacks
            verbose: Verbosity level

        Returns:
            Dictionary of training history
        """
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

            # Train
            train_metrics = []
            for batch in train_dataset:
                metrics = self.train_step(batch[0], batch[1])
                train_metrics.append(metrics)

            # Average metrics
            train_metrics = {
                k: tf.reduce_mean([m[k] for m in train_metrics])
                for k in train_metrics[0].keys()
            }

            # Validate
            if validation_dataset is not None:
                val_metrics = []
                val_predictions = []
                for batch in validation_dataset:
                    metrics, predictions = self.evaluate_step(batch[0], batch[1])
                    val_metrics.append(metrics)
                    val_predictions.append(predictions)

                val_metrics = {
                    k: tf.reduce_mean([m[k] for m in val_metrics])
                    for k in val_metrics[0].keys()
                }

            # Update history
            history['loss'].append(train_metrics['loss'])
            if validation_dataset is not None:
                history['val_loss'].append(val_metrics['loss'])
            for metric in self.metrics:
                history[metric.name].append(train_metrics[metric.name])
                if validation_dataset is not None:
                    history[f'val_{metric.name}'].append(val_metrics[metric.name])

            # Callbacks
            logs = {**train_metrics}
            if validation_dataset is not None:
                logs.update({f'val_{k}': v for k, v in val_metrics.items()})
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

            # Checkpointing
            if hasattr(self, 'ckpt_manager'):
                self.ckpt.step.assign_add(1)
                if int(self.ckpt.step) % 10 == 0:
                    self.ckpt_manager.save()

            # Print metrics
            if verbose:
                metrics_str = ' - '.join(
                    f'{k}: {v:.4f}' for k, v in logs.items()
                )
                print(metrics_str)

        return history

    """
    Additional training functionality implementations.
    """

    class CrossValidation:
        """
        K-fold cross-validation implementation for materials GNN models.

        Features:
        - Stratified splitting
        - Distributed training per fold
        - Advanced metrics aggregation
        - Memory-efficient processing

        Args:
            model_fn: Function to create model instance
            n_folds: Number of CV folds
            metrics: Metrics to track
            strategy: Distribution strategy
        """

        def __init__(
                self,
                model_fn: callable,
                n_folds: int = 5,
                metrics: Optional[List[str]] = None,
                strategy: Optional[tf.distribute.Strategy] = None
        ):
            self.model_fn = model_fn
            self.n_folds = n_folds
            self.metrics = metrics
            self.strategy = strategy or tf.distribute.get_strategy()

        def split_data(
                self,
                dataset: tf.data.Dataset,
                seed: int = 42
        ) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
            """Create stratified CV splits."""
            # Convert to numpy for splitting
            x_data = []
            y_data = []
            for x, y in dataset:
                x_data.append(x)
                y_data.append(y)
            x_data = np.array(x_data)
            y_data = np.array(y_data)

            # Create stratified folds
            fold_indices = []
            kf = tf.keras.utils.StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=seed
            )
            for train_idx, val_idx in kf.split(x_data, y_data):
                fold_indices.append((train_idx, val_idx))

            # Create datasets for each fold
            fold_datasets = []
            for train_idx, val_idx in fold_indices:
                train_data = tf.data.Dataset.from_tensor_slices(
                    (x_data[train_idx], y_data[train_idx])
                )
                val_data = tf.data.Dataset.from_tensor_slices(
                    (x_data[val_idx], y_data[val_idx])
                )
                fold_datasets.append((train_data, val_data))

            return fold_datasets

        def train_fold(
                self,
                train_data: tf.data.Dataset,
                val_data: tf.data.Dataset,
                fold: int,
                **train_kwargs
        ) -> Tuple[Dict[str, float], keras.Model]:
            """Train and evaluate model on a single fold."""
            with self.strategy.scope():
                # Create new model instance
                model = self.model_fn()
                trainer = TrainingEngine(
                    self.strategy,
                    model,
                    optimizer=train_kwargs.get('optimizer'),
                    loss_fn=train_kwargs.get('loss'),
                    metrics=self.metrics,
                    checkpoint_dir=f"fold_{fold}_checkpoints"
                )

                # Train model
                history = trainer.train(
                    train_data,
                    val_data,
                    epochs=train_kwargs.get('epochs', 100),
                    verbose=train_kwargs.get('verbose', 1)
                )

                # Get final metrics
                final_metrics = {
                    k: history[k][-1] for k in history.keys()
                }

            return final_metrics, model

        def run(
                self,
                dataset: tf.data.Dataset,
                **train_kwargs
        ) -> Tuple[Dict[str, List[float]], List[keras.Model]]:
            """Run complete cross-validation."""
            # Split data
            fold_datasets = self.split_data(dataset)

            # Train each fold
            fold_metrics = []
            fold_models = []
            for i, (train_data, val_data) in enumerate(fold_datasets):
                print(f"\nTraining fold {i + 1}/{self.n_folds}")
                metrics, model = self.train_fold(
                    train_data,
                    val_data,
                    i,
                    **train_kwargs
                )
                fold_metrics.append(metrics)
                fold_models.append(model)

            # Aggregate metrics
            aggregated_metrics = {}
            for key in fold_metrics[0].keys():
                values = [m[key] for m in fold_metrics]
                aggregated_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }

            return aggregated_metrics, fold_models

    class HyperparameterOptimization:
        """
        Advanced hyperparameter optimization for materials GNN models.

        Features:
        - Bayesian optimization
        - Multi-objective optimization
        - Distributed evaluation
        - Early stopping
        - Trial pruning

        Args:
            model_fn: Function to create model
            strategy: Distribution strategy
            metrics: Metrics to track
            search_space: Hyperparameter search space
        """

        def __init__(
                self,
                model_fn: callable,
                strategy: tf.distribute.Strategy,
                metrics: List[str],
                search_space: Dict[str, Any]
        ):
            self.model_fn = model_fn
            self.strategy = strategy
            self.metrics = metrics
            self.search_space = search_space

            # Set up keras tuner
            self.tuner = keras.tuners.BayesianOptimization(
                self._build_model,
                objective=keras.tuners.Objective(
                    'val_loss',
                    direction='min'
                ),
                max_trials=50,
                directory='hp_tuning',
                project_name='materials_gnn'
            )

        def _build_model(self, hp):
            """Build model with hyperparameters."""
            with self.strategy.scope():
                # Create model with hyperparameters
                hparams = {}
                for name, space in self.search_space.items():
                    if space['type'] == 'int':
                        hparams[name] = hp.Int(
                            name,
                            min_value=space['min'],
                            max_value=space['max'],
                            step=space.get('step', 1)
                        )
                    elif space['type'] == 'float':
                        hparams[name] = hp.Float(
                            name,
                            min_value=space['min'],
                            max_value=space['max']
                        )
                    elif space['type'] == 'choice':
                        hparams[name] = hp.Choice(
                            name,
                            values=space['values']
                        )

                model = self.model_fn(**hparams)
                return model

        def search(
                self,
                train_data: tf.data.Dataset,
                val_data: tf.data.Dataset,
                **train_kwargs
        ) -> Tuple[Dict[str, Any], keras.Model]:
            """Run hyperparameter search."""
            # Set up callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )
            ]

            # Run search
            self.tuner.search(
                train_data,
                validation_data=val_data,
                epochs=train_kwargs.get('epochs', 100),
                callbacks=callbacks
            )

            # Get best hyperparameters and model
            best_hp = self.tuner.get_best_hyperparameters()[0]
            best_model = self.tuner.get_best_models()[0]

            return best_hp.values, best_model

    class EnsembleTrainer:
        """
        Ensemble training for materials GNN models.

        Features:
        - Multi-model ensembling
        - Weighted averaging
        - Uncertainty estimation
        - Diversity metrics

        Args:
            model_fns: List of model creation functions
            strategy: Distribution strategy
            metrics: Metrics to track
        """

        def __init__(
                self,
                model_fns: List[callable],
                strategy: tf.distribute.Strategy,
                metrics: Optional[List[str]] = None
        ):
            self.model_fns = model_fns
            self.strategy = strategy
            self.metrics = metrics
            self.models = []

        def train_model(
                self,
                model_fn: callable,
                train_data: tf.data.Dataset,
                val_data: tf.data.Dataset,
                **train_kwargs
        ) -> Tuple[keras.Model, Dict[str, List[float]]]:
            """Train a single model."""
            with self.strategy.scope():
                model = model_fn()
                trainer = TrainingEngine(
                    self.strategy,
                    model,
                    optimizer=train_kwargs.get('optimizer'),
                    loss_fn=train_kwargs.get('loss'),
                    metrics=self.metrics
                )

                history = trainer.train(
                    train_data,
                    val_data,
                    epochs=train_kwargs.get('epochs', 100),
                    verbose=train_kwargs.get('verbose', 1)
                )

            return model, history

        def train_ensemble(
                self,
                train_data: tf.data.Dataset,
                val_data: tf.data.Dataset,
                **train_kwargs
        ) -> List[Tuple[keras.Model, Dict[str, List[float]]]]:
            """Train all models in ensemble."""
            results = []
            for i, model_fn in enumerate(self.model_fns):
                print(f"\nTraining model {i + 1}/{len(self.model_fns)}")
                model, history = self.train_model(
                    model_fn,
                    train_data,
                    val_data,
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
            for model in self.models:
                pred = model(inputs, training=False)
                predictions.append(pred)

            # Stack predictions
            predictions = tf.stack(predictions)

            # Calculate mean and std
            mean_pred = tf.reduce_mean(predictions, axis=0)
            std_pred = tf.math.reduce_std(predictions, axis=0)

            return mean_pred, std_pred

        def evaluate_ensemble(
                self,
                test_data: tf.data.Dataset
        ) -> Dict[str, float]:
            """Evaluate complete ensemble."""
            test_metrics = []
            for batch in test_data:
                mean_pred, std_pred = self.predict_ensemble(batch[0])

                # Calculate metrics
                metrics = {}
                metrics['loss'] = self.loss_fn(batch[1], mean_pred)
                metrics['uncertainty'] = tf.reduce_mean(std_pred)
                for metric in self.metrics:
                    metrics[metric.name] = metric(batch[1], mean_pred)

                test_metrics.append(metrics)

            # Average metrics
            final_metrics = {
                k: tf.reduce_mean([m[k] for m in test_metrics])
                for k in test_metrics[0].keys()
            }

            return final_metrics

    def setup_training(
            job_parameters: Dict[str, Any],
            model_parameters: Dict[str, Any]
    ) -> Tuple[tf.distribute.Strategy, Any]:
        """Set up training environment and strategy."""
        # Set up distribution strategy
        if len(tf.config.list_physical_devices('GPU')) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()

        # Set memory growth
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        # Set up logging
        if job_parameters.get('log_dir'):
            Path(job_parameters['log_dir']).mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                filename=f"{job_parameters['log_dir']}/training.log",
                level=logging.INFO
            )

        return strategy

    """
    Feature analysis and remaining training utilities.
    """

    class FeatureAnalysis:
        """
        Advanced feature analysis for materials GNN models.

        Features:
        - t-SNE visualization
        - PCA analysis
        - Feature importance
        - Attention visualization
        - Clustering analysis

        Args:
            model: Trained model to analyze
            strategy: Distribution strategy
            layer_names: Names of layers to analyze
        """

        def __init__(
                self,
                model: keras.Model,
                strategy: tf.distribute.Strategy,
                layer_names: Optional[List[str]] = None
        ):
            self.model = model
            self.strategy = strategy
            self.layer_names = layer_names or []

            # Set up feature extractors
            self.extractors = {}
            for name in self.layer_names:
                layer = self.model.get_layer(name)
                self.extractors[name] = keras.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )

        def extract_features(
                self,
                dataset: tf.data.Dataset,
                layer_name: str
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Extract features from specific layer."""
            features = []
            labels = []

            extractor = self.extractors[layer_name]
            for x, y in dataset:
                feat = extractor(x, training=False)
                features.append(feat)
                labels.append(y)

            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)

            return features, labels

        def run_tsne(
                self,
                features: np.ndarray,
                n_components: int = 2,
                perplexity: float = 30.0,
                learning_rate: float = 'auto',
                **kwargs
        ) -> np.ndarray:
            """Run t-SNE dimensionality reduction."""
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                **kwargs
            )
            transformed = tsne.fit_transform(features)
            return transformed

        def run_pca(
                self,
                features: np.ndarray,
                n_components: float = 0.95
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Run PCA analysis."""
            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(features)
            return transformed, pca.components_, pca.explained_variance_ratio_

        def analyze_attention(
                self,
                dataset: tf.data.Dataset,
                layer_name: str
        ) -> Dict[str, np.ndarray]:
            """Analyze attention weights."""
            attention_weights = []
            node_features = []

            # Get attention layer
            layer = self.model.get_layer(layer_name)

            # Extract attention weights
            for x, _ in dataset:
                weights = layer(x, training=False)
                attention_weights.append(weights)
                node_features.append(x)

            attention_weights = np.concatenate(attention_weights, axis=0)
            node_features = np.concatenate(node_features, axis=0)

            return {
                'weights': attention_weights,
                'features': node_features
            }

        def feature_importance(
                self,
                dataset: tf.data.Dataset,
                n_permutations: int = 100
        ) -> Dict[str, np.ndarray]:
            """Calculate feature importance via permutation."""
            baseline_metrics = self.model.evaluate(dataset)
            importance_scores = {}

            # For each feature
            for i in range(self.model.input_shape[-1]):
                scores = []

                # Permute feature multiple times
                for _ in range(n_permutations):
                    permuted_data = dataset.map(
                        lambda x, y: (self._permute_feature(x, i), y)
                    )
                    metrics = self.model.evaluate(permuted_data)
                    scores.append(metrics['loss'] - baseline_metrics['loss'])

                importance_scores[f'feature_{i}'] = np.mean(scores)

            return importance_scores

        @tf.function
        def _permute_feature(
                self,
                x: tf.Tensor,
                feature_idx: int
        ) -> tf.Tensor:
            """Permute a single feature."""
            permuted = tf.random.shuffle(x[..., feature_idx])
            return tf.tensor_scatter_nd_update(
                x,
                tf.expand_dims(tf.range(tf.shape(x)[0]), 1),
                permuted
            )

        def visualize_clusters(
                self,
                features: np.ndarray,
                labels: np.ndarray,
                n_clusters: int = 5
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Analyze feature clusters."""
            from sklearn.cluster import KMeans

            # Run clustering
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(features)

            # Calculate cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                mask = clusters == i
                cluster_stats[f'cluster_{i}'] = {
                    'size': np.sum(mask),
                    'mean_label': np.mean(labels[mask]),
                    'std_label': np.std(labels[mask])
                }

            return clusters, cluster_stats

    def setup_distributed_training(
            num_gpus: int,
            job_parameters: Dict[str, Any]
    ) -> tf.distribute.Strategy:
        """
        Set up distributed training strategy.

        Args:
            num_gpus: Number of GPUs to use
            job_parameters: Training configuration

        Returns:
            Distribution strategy
        """
        # Set up strategy based on available GPUs
        if num_gpus > 1:
            if job_parameters.get('use_horovod', False):
                import horovod.tensorflow as hvd
                hvd.init()
                strategy = tf.distribute.HorovodStrategy()
            else:
                strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0" if num_gpus == 1 else "/cpu:0")

        return strategy

    class ModelCheckpointing:
        """
        Advanced model checkpointing with features for materials models.

        Features:
        - Best model tracking
        - Periodic saving
        - State restoration
        - Memory efficient saving

        Args:
            model: Model to checkpoint
            directory: Checkpoint directory
            monitor: Metric to monitor
            save_best_only: Whether to save only best models
        """

        def __init__(
                self,
                model: keras.Model,
                directory: str,
                monitor: str = 'val_loss',
                save_best_only: bool = True
        ):
            self.model = model
            self.directory = Path(directory)
            self.monitor = monitor
            self.save_best_only = save_best_only

            self.directory.mkdir(parents=True, exist_ok=True)
            self.best_value = float('inf')

        def save(
                self,
                epoch: int,
                metrics: Dict[str, float]
        ):
            """Save model checkpoint."""
            current = metrics.get(self.monitor, float('inf'))

            if self.save_best_only:
                if current < self.best_value:
                    self._save_checkpoint(epoch, metrics)
                    self.best_value = current
            else:
                self._save_checkpoint(epoch, metrics)

        def _save_checkpoint(
                self,
                epoch: int,
                metrics: Dict[str, float]
        ):
            """Internal save function."""
            # Save weights
            weights_path = self.directory / f"weights_{epoch:03d}.h5"
            self.model.save_weights(weights_path)

            # Save optimizer state
            optimizer_path = self.directory / f"optimizer_{epoch:03d}.npy"
            np.save(optimizer_path, self.model.optimizer.get_weights())

            # Save metrics
            metrics_path = self.directory / f"metrics_{epoch:03d}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)

        def restore(
                self,
                epoch: Optional[int] = None
        ):
            """Restore model from checkpoint."""
            if epoch is None:
                # Find latest checkpoint
                checkpoints = list(self.directory.glob("weights_*.h5"))
                if not checkpoints:
                    return
                latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
                epoch = int(latest.stem.split('_')[1])

            # Restore weights
            weights_path = self.directory / f"weights_{epoch:03d}.h5"
            self.model.load_weights(weights_path)

            # Restore optimizer state
            optimizer_path = self.directory / f"optimizer_{epoch:03d}.npy"
            if optimizer_path.exists():
                optimizer_weights = np.load(optimizer_path, allow_pickle=True)
                self.model.optimizer.set_weights(optimizer_weights)

    def write_results(
            outputs: np.ndarray,
            filename: str,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """Write model outputs with metadata."""
        # Create output directory
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Write outputs
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write metadata
            if metadata:
                for key, value in metadata.items():
                    writer.writerow([f"# {key}", value])
                writer.writerow([])

            # Write header
            writer.writerow(['id', 'target'] + [f'pred_{i}' for i in range(outputs.shape[1] - 2)])

            # Write data
            writer.writerows(outputs)




