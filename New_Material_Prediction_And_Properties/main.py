import os
import argparse
import time
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
from qiskit import Aer, execute, QuantumCircuit
from sklearn.metrics import mean_absolute_error, r2_score

from matdeeplearn import models, process, training
from matdeeplearn.training import TrainingConfig, configure_training
from matdeeplearn.utils.hardware import get_distribution_strategy
from matdeeplearn.utils.typing import Dataset, Model
from matdeeplearn.models import (
    CGCNN, DeepGATGNN, SOAP, SM, GATGNN, GCN, MEGNet, MPNN, SchNet,
    SuperCGCNN, SuperMEGNet, SuperMPNN, SuperSchNet, QuantumGAN
)


class MaterialsLearner:
    """Main controller for materials deep learning."""

    def __init__(self):
        self.config = None
        self.strategy = None
        self.logger = self._setup_logging()
        self.models = {}

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('MatDeepLearn')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def setup_environment(self, mixed_precision: bool = True) -> None:
        self.strategy = get_distribution_strategy()
        self.logger.info(f"Using strategy: {self.strategy.__class__.__name__}")

        if mixed_precision and tf.config.list_physical_devices('GPU'):
            set_global_policy('mixed_float16')

        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

    def _initialize_models(self) -> None:
        model_classes = {
            'cgcnn': CGCNN,
            'deep_gatgnn': DeepGATGNN,
            'soap': SOAP,
            'sm': SM,
            'gatgnn': GATGNN,
            'gcn': GCN,
            'megnet': MEGNet,
            'mpnn': MPNN,
            'schnet': SchNet,
            'super_cgcnn': SuperCGCNN,
            'super_megnet': SuperMEGNet,
            'super_mpnn': SuperMPNN,
            'super_schnet': SuperSchNet,
            'quantum_gan': QuantumGAN
        }

        for model_name, model_cls in model_classes.items():
            if model_name in self.config['Models']:
                with self.strategy.scope():
                    self.models[model_name] = model_cls(**self.config['Models'][model_name])

    def load_data(self) -> Dataset:
        return process.get_dataset(
            data_path=Path("data"),
            target_index=self.config['Training']['target_index'],
            reprocess=self.config['Job'].get('reprocess', False),
            processing_config=self.config['Processing']
        )

    def train_models(self, dataset: Dataset) -> Dict[str, Any]:
        results = {}
        training_config = configure_training(self.config['Training'])

        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}")
            try:
                if model_name == 'quantum_gan':
                    results[model_name] = self._train_quantum_model(model, dataset)
                else:
                    results[model_name] = self._train_classical_model(
                        model, dataset, training_config
                    )
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue

        return results

    def _train_classical_model(self,
                               model: tf.keras.Model,
                               dataset: Dataset,
                               training_config: TrainingConfig) -> Dict[str, Any]:
        trainer = training.Trainer(model, training_config, self.strategy)
        callbacks = self._create_callbacks(model.__class__.__name__)

        train_data, val_data, test_data = process.split_dataset(
            dataset=dataset,
            **self.config['Training']['split_ratios']
        )

        history = trainer.fit(train_data, val_data, callbacks)
        test_results = trainer.evaluate(test_data) if test_data else None

        return {
            'history': history,
            'test_results': test_results
        }

    def _train_quantum_model(self,
                             model: QuantumGAN,
                             dataset: Dataset) -> Dict[str, Any]:
        quantum_config = self.config['Models']['quantum_gan']
        train_data, val_data, _ = process.split_dataset(
            dataset=dataset,
            **self.config['Training']['split_ratios']
        )

        history = {'d_loss': [], 'g_loss': []}
        for epoch in range(quantum_config['epochs']):
            d_loss, g_loss = model.train_step(train_data)
            history['d_loss'].append(d_loss)
            history['g_loss'].append(g_loss)

            if epoch % 10 == 0:
                val_results = model.evaluate(val_data)
                self.logger.info(f"Epoch {epoch}: {val_results}")

        return {'history': history}

    def _create_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                f"checkpoints/{model_name}/best_model.h5",
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(**self.config['Training']['early_stopping']),
            tf.keras.callbacks.ReduceLROnPlateau(**self.config['Training']['reduce_lr']),
            tf.keras.callbacks.TensorBoard(log_dir=f"logs/{model_name}")
        ]

    def save_results(self, results: Dict[str, Any]) -> None:
        save_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_results in results.items():
            with open(save_dir / f"{model_name}_results.json", 'w') as f:
                json.dump(model_results, f, indent=4)

            if model_name != 'quantum_gan':
                self.models[model_name].save(save_dir / f"{model_name}_model")

    def run(self, config_path: str) -> None:
        try:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

            self.setup_environment()
            self._initialize_models()
            dataset = self.load_data()
            results = self.train_models(dataset)
            self.save_results(results)

        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}")
            raise


def main():
    start_time = time.time()

    try:
        learner = MaterialsLearner()
        learner.run("config.yml")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

    finally:
        elapsed_time = time.time() - start_time
        logging.info(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
