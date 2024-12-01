import tensorflow as tf
import yaml
import mlflow
from pathlib import Path
import logging
from datetime import datetime

from models import CGCNN, MEGNet, SchNet, SuperCGCNN, SuperSchNet
from process import DataProcessor
from training import Trainer
from data import MunicipalRoadDataGenerator
from utils.callbacks import SaveBestModel, EarlyStoppingWithRestore
from utils.metrics import evaluate_model
from utils.visualization import plot_history, plot_predictions


def setup_environment():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Generate data if needed
    data_path = Path(config['data_path'])
    if not data_path.exists():
        logger.info("Generating municipal-grade dataset...")
        generator = MunicipalRoadDataGenerator()
        generator.generate_dataset(data_path)

    # Initialize data processor
    processor = DataProcessor(config['data'])
    train_data, val_data, test_data = processor.load_data(data_path)

    # Create TensorFlow datasets
    train_ds = processor.create_tf_dataset(train_data, config['batch_size'])
    val_ds = processor.create_tf_dataset(val_data, config['batch_size'])
    test_ds = processor.create_tf_dataset(test_data, config['batch_size'])

    # Initialize models
    models = {
        'cgcnn': CGCNN(config['model']['cgcnn']),
        'megnet': MEGNet(config['model']['megnet']),
        'schnet': SchNet(config['model']['schnet']),
        'super_cgcnn': SuperCGCNN(config['model']['super_cgcnn']),
        'super_schnet': SuperSchNet(config['model']['super_schnet'])
    }

    # Training loop for each model
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        mlflow.set_experiment(f"road_materials_{name}")

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(config['model'][name])

            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
                loss='mse',
                metrics=['mae', 'mse']
            )

            # Training callbacks
            callbacks = [
                SaveBestModel(f'checkpoints/{name}_best.h5'),
                EarlyStoppingWithRestore(patience=10),
                tf.keras.callbacks.TensorBoard(log_dir=f'logs/{name}_{timestamp}')
            ]

            # Train model
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=config['epochs'],
                callbacks=callbacks
            )

            # Evaluate model
            test_metrics = evaluate_model(model, test_ds, uncertainty=True)
            results[name] = test_metrics

            # Log metrics
            mlflow.log_metrics(test_metrics)

            # Generate plots
            plot_dir = Path('plots')
            plot_dir.mkdir(exist_ok=True)

            plot_history(history.history,
                         output_path=plot_dir / f'{name}_history.png')
            mlflow.log_artifact(plot_dir / f'{name}_history.png')

            # Save model
            model.save(f'models/{name}_{timestamp}')
            mlflow.log_artifact(f'models/{name}_{timestamp}')

    # Compare model results
    logger.info("\nModel Comparison:")
    for name, metrics in results.items():
        logger.info(f"\n{name}:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    # Save final results
    import json
    with open(f'results/comparison_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    setup_environment()
    main()