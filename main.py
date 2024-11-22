import os
import time
import sys
import json
import yaml
import boto3
import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tensorflow.distribute import MirroredStrategy
from ray import tune

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, config):
        self.config = config
        self.s3_client = boto3.client('s3') if config.get('use_s3', False) else None

    def load_dataset(self):
        data = {}

        # Load from local data directory
        if os.path.exists('data'):
            data.update(self._load_local_data())

        # Load from S3 if configured
        if self.s3_client:
            s3_data = self._load_s3_data()
            data.update(self._merge_data(data, s3_data))

        return self._prepare_tf_datasets(data)

    def _load_local_data(self):
        data = {}
        data_dir = Path('data')
        for file in data_dir.glob('*.npy'):
            data[file.stem] = np.load(file)
        return data

    def _load_s3_data(self):
        data = {}
        try:
            response = self.s3_client.list_objects_v2(
                Bucket='concrete-aws',
                Prefix='New_Material_Properties_And_Predictions'
            )
            for obj in response['Contents']:
                if obj['Key'].endswith('.npy'):
                    file_data = self.s3_client.get_object(
                        Bucket='concrete-aws',
                        Key=obj['Key']
                    )
                    data[Path(obj['Key']).stem] = np.load(file_data['Body'])
        except Exception as e:
            logger.error(f"Error loading S3 data: {e}")
        return data

    def _prepare_tf_datasets(self, data):
        # Create TF datasets for training
        features = data['features']
        targets = data['targets']

        dataset = tf.data.Dataset.from_tensor_slices((features, targets))
        dataset = dataset.shuffle(10000).batch(self.config['batch_size'])

        # Split into train/val/test
        total_size = len(features)
        train_size = int(total_size * self.config['train_ratio'])
        val_size = int(total_size * self.config['val_ratio'])

        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size).take(val_size)
        test_ds = dataset.skip(train_size + val_size)

        return {'train': train_ds, 'val': val_ds, 'test': test_ds}


class Trainer:
    def __init__(self, config):
        self.config = config
        self.strategy = self._setup_strategy()

    def _setup_strategy(self):
        if len(tf.config.list_physical_devices('GPU')) > 1:
            return MirroredStrategy()
        return tf.distribute.get_strategy()

    def train(self, model_type, datasets):
        logger.info(f"Training {model_type}")

        with self.strategy.scope():
            model = self._build_model(model_type)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(self.config['learning_rate']),
                loss=self.config['loss'],
                metrics=['mae', 'mse']
            )

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/{model_type}_best.h5',
                    save_best_only=True
                ),
                tf.keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=f'logs/{model_type}'
                )
            ]

            history = model.fit(
                datasets['train'],
                validation_data=datasets['val'],
                epochs=self.config['epochs'],
                callbacks=callbacks
            )

            self._save_results(history.history, model_type)

    def _build_model(self, model_type):
        from models import (
            SchNetModel, SuperSchNetModel,
            MPNNModel, SuperMPNNModel,
            CGCNNModel, SuperCGCNNModel,
            MEGNetModel, SuperMEGNetModel,
            GCNModel,
            GATGNNModel, DeepGATGNNModel,
            SOAPModel, SMModel,
            VAEModel, CycleGANModel,
            DescriptorNNModel, QuantumGANModel
        )

        builders = {
            'SchNet_demo': SchNetModel,
            'Super_SchNet_demo': SuperSchNetModel,
            'MPNN_demo': MPNNModel,
            'Super_MPNN_demo': SuperMPNNModel,
            'CGCNN_demo': CGCNNModel,
            'Super_CGCNN_demo': SuperCGCNNModel,
            'MEGNet_demo': MEGNetModel,
            'Super_MEGNet_demo': SuperMEGNetModel,
            'GCN_demo': GCNModel,
            'GATGNN_demo': GATGNNModel,
            'DEEP_GATGNN_demo': DeepGATGNNModel,
            'SOAP_demo': SOAPModel,
            'SM_demo': SMModel,
            'VAE_demo': VAEModel,
            'Cycle_GAN_demo': CycleGANModel,
            'Descriptor_NN_demo': DescriptorNNModel,
            'Quantum_GAN_demo': QuantumGANModel
        }

        builder = builders.get(model_type)
        if not builder:
            raise ValueError(f"Unknown model type: {model_type}")

        return builder(self.config)

    def _save_results(self, history, model_type):
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(results_dir / f'{model_type}_{timestamp}.json', 'w') as f:
            json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'cv', 'hyperparameter'])
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup directories
    for dirname in ['models', 'logs', 'results']:
        Path(dirname).mkdir(exist_ok=True)

    # Load data
    data_handler = DataHandler(config)
    datasets = data_handler.load_dataset()

    # Initialize trainer
    trainer = Trainer(config)

    # Run based on mode
    if args.mode == 'train':
        trainer.train(args.model, datasets)
    elif args.mode == 'hyperparameter':
        # Configure Ray Tune
        tune_config = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "hidden_units": tune.choice([64, 128, 256])
        }

        analysis = tune.run(
            lambda config: trainer.train(args.model, datasets),
            config=tune_config,
            num_samples=50
        )

        print("Best hyperparameters:", analysis.best_config)


if __name__ == "__main__":
    main()
