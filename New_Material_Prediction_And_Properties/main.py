import os
import argparse
import time
import sys
import json
import pprint
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import tensorflow as tf
import numpy as np
from tensorflow.keras.mixed_precision import set_global_policy

from matdeeplearn import models, process, training
from matdeeplearn.training import TrainingConfig, configure_training
from matdeeplearn.utils.hardware import get_distribution_strategy
from matdeeplearn.utils.typing import Dataset, Model


class MaterialsLearner:
    """Main controller class for materials deep learning."""

    def __init__(self):
        self.config = None
        self.strategy = None
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging with appropriate format."""
        logger = logging.getLogger('MatDeepLearn')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def setup_environment(self, mixed_precision: bool = True) -> None:
        """Initialize training environment and hardware."""
        # Set up distribution strategy
        self.strategy = get_distribution_strategy()
        self.logger.info(f"Using distribution strategy: {self.strategy.__class__.__name__}")

        # Configure precision
        if mixed_precision and tf.config.list_physical_devices('GPU'):
            set_global_policy('mixed_float16')
            self.logger.info("Enabled mixed precision training")

        # Set memory growth
        for gpu in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                self.logger.warning(f"Memory growth setting failed: {e}")

    def parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command line arguments."""
        parser = argparse.ArgumentParser(description="MatDeepLearn TensorFlow")

        # Add argument groups for better organization
        job_args = parser.add_argument_group('Job Configuration')
        job_args.add_argument("--config_path", default="config.yml", type=str)
        job_args.add_argument(
            "--run_mode",
            choices=[
                "Training", "Predict", "Repeat", "CV",
                "Hyperparameter", "Ensemble", "Analysis"
            ],
            default=None
        )
        job_args.add_argument("--job_name", default=None, type=str)
        job_args.add_argument("--model", default=None, type=str)
        job_args.add_argument("--seed", default=None, type=int)
        job_args.add_argument("--model_path", default=None, type=str)

        # Data processing arguments
        proc_args = parser.add_argument_group('Data Processing')
        proc_args.add_argument("--data_path", default=None, type=str)
        proc_args.add_argument("--format", default=None, type=str)
        proc_args.add_argument("--file_name", default=None, type=str)
        proc_args.add_argument("--reprocess", action="store_true")

        # Training arguments
        train_args = parser.add_argument_group('Training')
        train_args.add_argument("--train_ratio", default=None, type=float)
        train_args.add_argument("--val_ratio", default=None, type=float)
        train_args.add_argument("--test_ratio", default=None, type=float)
        train_args.add_argument("--verbosity", default=None, type=int)
        train_args.add_argument("--target_index", default=None, type=int)
        train_args.add_argument("--batch_size", default=None, type=int)
        train_args.add_argument("--lr", default=None, type=float)
        train_args.add_argument("--epochs", default=None, type=int)

        args = parser.parse_args()
        self._validate_arguments(args)
        return args

    def _validate_arguments(self, args: argparse.Namespace) -> None:
        """Validate command line arguments."""
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(f"Config file not found: {args.config_path}")

        if args.train_ratio is not None:
            if not 0 < args.train_ratio < 1:
                raise ValueError("Train ratio must be between 0 and 1")

        # Similar validation for other ratios and numeric parameters

    def load_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Load and merge configuration from file and command line."""
        with open(args.config_path) as f:
            config = yaml.safe_load(f)

        # Update config with command line arguments
        self._update_config(config, args)

        # Validate final configuration
        self._validate_config(config)

        return config

    def _update_config(self, config: Dict[str, Any], args: argparse.Namespace) -> None:
        """Update configuration with command line arguments."""
        if args.run_mode:
            config['Job']['run_mode'] = args.run_mode

        run_mode = config['Job']['run_mode']
        if run_mode not in config['Job']:
            raise ValueError(f"Configuration for run mode {run_mode} not found")

        mode_config = config['Job'][run_mode]

        # Update various configuration sections
        self._update_job_config(mode_config, args)
        self._update_processing_config(config['Processing'], args)
        self._update_training_config(config['Training'], args)
        self._update_model_config(config['Models'], args)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate complete configuration."""
        required_sections = ['Job', 'Processing', 'Training', 'Models']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate job configuration
        job_config = config['Job']
        if 'run_mode' not in job_config:
            raise ValueError("Run mode not specified in configuration")

        # Additional validation based on run mode

    def run(self, args: argparse.Namespace) -> None:
        """Main execution flow."""
        try:
            # Load and validate configuration
            self.config = self.load_config(args)

            # Setup training environment
            self.setup_environment(
                mixed_precision=self.config['Training'].get('mixed_precision', True)
            )

            # Initialize training configuration
            training_config = configure_training(self.config['Training'])

            # Load or process dataset
            dataset = self._prepare_dataset()

            # Execute requested run mode
            run_mode = self.config['Job']['run_mode']
            self.logger.info(f"Executing run mode: {run_mode}")

            if run_mode == "Training":
                self._run_training(dataset, training_config)
            elif run_mode == "Predict":
                self._run_prediction(dataset)
            elif run_mode == "CV":
                self._run_cross_validation(dataset, training_config)
            elif run_mode == "Repeat":
                self._run_repeated_training(dataset, training_config)
            elif run_mode == "Hyperparameter":
                self._run_hyperparameter_optimization(training_config)
            elif run_mode == "Ensemble":
                self._run_ensemble_training(dataset, training_config)
            elif run_mode == "Analysis":
                self._run_analysis(dataset)

        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}")
            raise

    def _prepare_dataset(self) -> Dataset:
        """Prepare dataset based on configuration."""
        if self.config['Job']['run_mode'] != "Hyperparameter":
            dataset = process.get_dataset(
                self.config['Processing']['data_path'],
                self.config['Training']['target_index'],
                self.config['Job'].get('reprocess', False),
                self.config['Processing']
            )
            self.logger.info(f"Dataset prepared: {dataset}")
            return dataset

        return None

    # Individual run mode implementations...
    # (I'll continue with these in the next message)


def main():
    """Entry point for the application."""
    start_time = time.time()

    try:
        learner = MaterialsLearner()
        args = learner.parse_arguments()
        learner.run(args)

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

    finally:
        elapsed_time = time.time() - start_time
        logging.info(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

def _run_training(self,
                  dataset: Dataset,
                  training_config: TrainingConfig) -> None:
    """Execute training run with comprehensive monitoring and checkpointing.

    Args:
        dataset: Prepared dataset
        training_config: Training configuration
    """
    self.logger.info("Starting training run")

    # Create model within strategy scope
    with self.strategy.scope():
        # Build model
        model = self._create_model()

        # Set up trainer
        trainer = training.Trainer(
            model=model,
            config=training_config,
            strategy=self.strategy
        )

        # Prepare datasets
        train_data, val_data, test_data = process.split_dataset(
            dataset=dataset,
            train_ratio=self.config['Training']['train_ratio'],
            val_ratio=self.config['Training']['val_ratio'],
            test_ratio=self.config['Training']['test_ratio'],
            seed=self.config['Job'].get('seed', None)
        )

        # Set up callbacks
        callbacks = self._create_callbacks()

        # Train model
        history = trainer.fit(
            train_data=train_data,
            validation_data=val_data,
            callbacks=callbacks
        )

        # Evaluate on test set
        if test_data is not None:
            test_results = trainer.evaluate(test_data)
            self.logger.info(f"Test set results: {test_results}")

        # Save results
        self._save_training_results(
            trainer=trainer,
            history=history,
            test_results=test_results if test_data is not None else None
        )


def _create_model(self) -> Model:
    """Create model instance based on configuration.

    Returns:
        Instantiated model
    """
    model_name = self.config['Job'].get('model')
    model_config = self.config['Models'][model_name]

    model_cls = getattr(models, model_name)
    return model_cls(**model_config)


def _create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
    """Create training callbacks.

    Returns:
        List of callbacks
    """
    callbacks = []

    # Model checkpoint
    if self.config['Job'].get('save_model'):
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config['Job']['model_path'],
                save_best_only=True,
                monitor='val_loss'
            )
        )

    # Early stopping
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['Training'].get('early_stopping_patience', 10),
            restore_best_weights=True
        )
    )

    # Learning rate scheduler
    if 'scheduler' in self.config['Training']:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                **self.config['Training']['scheduler']
            )
        )

    # TensorBoard logging
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=f"logs/{self.config['Job']['job_name']}",
            histogram_freq=1
        )
    )

    return callbacks


def _save_training_results(self,
                           trainer: training.Trainer,
                           history: Dict[str, List[float]],
                           test_results: Optional[Dict[str, float]] = None) -> None:
    """Save training results and model.

    Args:
        trainer: Training manager instance
        history: Training history
        test_results: Optional test set results
    """
    output_dir = Path(self.config['Job']['job_name'])
    output_dir.mkdir(exist_ok=True)

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    # Save test results if available
    if test_results is not None:
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4)

    # Save model if requested
    if self.config['Job'].get('save_model'):
        model_path = output_dir / 'model'
        trainer.save(model_path)
        self.logger.info(f"Model saved to {model_path}")


def _run_prediction(self, dataset: Dataset) -> None:
    """Run prediction using trained model.

    Args:
        dataset: Dataset to predict on
    """
    self.logger.info("Starting prediction run")

    # Load model
    model_path = self.config['Job']['model_path']
    with self.strategy.scope():
        model = tf.keras.models.load_model(model_path)

    # Make predictions
    predictions = model.predict(dataset)

    # Save predictions
    if self.config['Job'].get('write_output'):
        output_path = Path(self.config['Job']['job_name']) / 'predictions.csv'
        self._save_predictions(predictions, output_path)


def _run_cross_validation(self,
                          dataset: Dataset,
                          training_config: TrainingConfig) -> None:
    """Execute cross-validation study.

    Args:
        dataset: Input dataset
        training_config: Training configuration
    """
    self.logger.info("Starting cross-validation")

    cv_config = self.config['Job']['CV']
    n_folds = cv_config.get('cv_folds', 5)

    # Create cross-validator
    cv = training.CrossValidator(
        model_builder=self._create_model,
        n_splits=n_folds,
        config=training_config
    )

    # Run cross-validation
    results = cv.cross_validate(dataset)

    # Save results
    self._save_cv_results(results)


def _run_repeated_training(self,
                           dataset: Dataset,
                           training_config: TrainingConfig) -> None:
    """Execute multiple training runs for statistical analysis.

    Args:
        dataset: Input dataset
        training_config: Training configuration
    """
    self.logger.info("Starting repeated training")

    n_trials = self.config['Job']['repeat_trials']
    results = []

    for trial in range(n_trials):
        self.logger.info(f"Starting trial {trial + 1}/{n_trials}")

        # Set new seed for this trial
        seed = self.config['Job'].get('seed', 0) + trial
        tf.random.set_seed(seed)

        # Create new model and trainer
        with self.strategy.scope():
            model = self._create_model()
            trainer = training.Trainer(model, training_config, self.strategy)

            # Train model
            history = trainer.fit(dataset)
            results.append(history)

    # Analyze results
    self._analyze_repeated_results(results)


def _run_hyperparameter_optimization(self,
                                     training_config: TrainingConfig) -> None:
    """Execute hyperparameter optimization study.

    Args:
        training_config: Base training configuration
    """
    self.logger.info("Starting hyperparameter optimization")

    # Set up optimization
    optimizer = training.HyperParameterOptimizer(
        model_fn=self._create_model,
        param_space=self._get_param_space(),
        optimization_metric='val_loss',
        max_trials=self.config['Job']['hyper_trials']
    )

    # Run optimization
    best_params = optimizer.optimize(training_config)

    # Save results
    self._save_hyperparameter_results(best_params)


def _run_ensemble_training(self,
                           dataset: Dataset,
                           training_config: TrainingConfig) -> None:
    """Train ensemble of models.

    Args:
        dataset: Input dataset
        training_config: Training configuration
    """
    self.logger.info("Starting ensemble training")

    # Create model builders
    model_builders = [
        self._create_model_builder(model_name)
        for model_name in self.config['Job']['ensemble_list']
    ]

    # Create ensemble trainer
    ensemble = training.EnsembleTrainer(
        model_builders=model_builders,
        config=training_config
    )

    # Train ensemble
    history = ensemble.fit(dataset)

    # Save results
    self._save_ensemble_results(history)

## Advanced Usage

### Custom Models
from matdeeplearn.models import BaseModel


class CustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Model initialization


### Custom Training
from matdeeplearn.training import Trainer


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom training logic


