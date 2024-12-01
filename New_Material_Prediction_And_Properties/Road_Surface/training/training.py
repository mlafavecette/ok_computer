import tensorflow as tf
from pathlib import Path
import logging
from typing import Dict
import yaml
import mlflow
from datetime import datetime

from model.model import RoadMaterialsModel, ModelConfig
from process.process import DataProcessor


class Trainer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_mlflow()

    def _load_config(self, path: str) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/training_{datetime.now():%Y%m%d_%H%M%S}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_mlflow(self):
        mlflow.set_experiment(self.config['experiment_name'])

    def train(self):
        # Initialize processor and load data
        processor = DataProcessor(self.config['data'])
        train_data, val_data, test_data = processor.load_data(
            self.config['data_path']
        )

        # Create datasets
        train_ds = processor.create_tf_dataset(
            train_data, self.config['batch_size']
        )
        val_ds = processor.create_tf_dataset(
            val_data, self.config['batch_size']
        )

        # Initialize and compile model
        model = RoadMaterialsModel(ModelConfig(**self.config['model']))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        # Training
        with mlflow.start_run():
            mlflow.log_params(self.config['model'])

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config['epochs'],
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        'checkpoints/best_model.h5',
                        save_best_only=True
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )

            # Log metrics
            for metric, values in history.history.items():
                mlflow.log_metric(metric, values[-1])

            # Evaluate on test set
            test_ds = processor.create_tf_dataset(
                test_data, self.config['batch_size']
            )
            test_results = model.evaluate(test_ds)

            for metric, value in zip(model.metrics_names, test_results):
                mlflow.log_metric(f'test_{metric}', value)

        return model, history