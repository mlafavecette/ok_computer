import tensorflow as tf
import wandb
from typing import Dict, Optional
from pathlib import Path
import json
import logging
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 100
    early_stopping_patience: int = 10


class MaterialsTrainer:
    """Training pipeline for materials property prediction."""

    def __init__(
            self,
            model: tf.keras.Model,
            config: TrainingConfig,
            experiment_name: str,
            checkpoint_dir: Path
    ):
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.setup_training()
        self.setup_logging(experiment_name)

    def setup_training(self):
        """Initialize optimizer and metrics."""
        self.optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)
        self.train_loss = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()

    def setup_logging(self, experiment_name: str):
        """Configure logging and experiment tracking."""
        wandb.init(project="road-materials", name=experiment_name)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @tf.function
    def train_step(self, batch: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Execute single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(batch, training=True)
            loss = self.compute_loss(predictions, batch['targets'])

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        return loss

    @tf.function
    def val_step(self, batch: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Execute single validation step."""
        predictions = self.model(batch, training=False)
        loss = self.compute_loss(predictions, batch['targets'])
        self.val_loss.update_state(loss)
        return loss

    def train(
            self,
            train_data: tf.data.Dataset,
            val_data: tf.data.Dataset
    ):
        """Execute training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            # Training epoch
            for batch in train_data:
                self.train_step(batch)

            # Validation epoch
            for batch in val_data:
                self.val_step(batch)

            # Log metrics
            metrics = {
                'train_loss': self.train_loss.result(),
                'val_loss': self.val_loss.result(),
                'epoch': epoch
            }
            self.log_metrics(metrics)

            # Early stopping
            val_loss = self.val_loss.result()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, metrics)
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            # Reset metrics
            self.train_loss.reset_states()
            self.val_loss.reset_states()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}"
        self.model.save(checkpoint_path)

        metadata_path = self.checkpoint_dir / f"metadata_epoch_{epoch}.json"
        with open(metadata_path, 'w') as f:
            json.dump({'epoch': epoch, 'metrics': metrics}, f)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tracking system."""
        wandb.log(metrics)
        self.logger.info(f"Metrics: {metrics}")


class VAETrainer(MaterialsTrainer):
    """Specialized trainer for VAE models."""

    def compute_loss(
            self,
            outputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            targets: tf.Tensor
    ) -> tf.Tensor:
        reconstruction, mean, logvar = outputs

        # Reconstruction loss
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(targets - reconstruction), axis=-1)
        )

        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
        )

        return recon_loss + kl_loss


class CycleGANTrainer(MaterialsTrainer):
    """Specialized trainer for CycleGAN models."""

    def setup_training(self):
        """Initialize optimizers for generators and discriminators."""
        self.gen_optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)
        self.disc_optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)

    @tf.function
    def train_step(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Execute single training step for CycleGAN."""
        real_A, real_B = batch['A'], batch['B']

        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images
            fake_B = self.model.gen_A2B(real_A)
            fake_A = self.model.gen_B2A(real_B)

            # Cycle consistency
            cycle_A = self.model.gen_B2A(fake_B)
            cycle_B = self.model.gen_A2B(fake_A)

            # Generator losses
            gen_loss = self._compute_generator_loss(
                real_A, real_B, fake_A, fake_B, cycle_A, cycle_B
            )

            # Discriminator losses
            disc_loss = self._compute_discriminator_loss(
                real_A, real_B, fake_A, fake_B
            )

        # Update generators
        gen_gradients = tape.gradient(
            gen_loss,
            self.model.gen_A2B.trainable_variables + self.model.gen_B2A.trainable_variables
        )
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients,
                self.model.gen_A2B.trainable_variables + self.model.gen_B2A.trainable_variables)
        )

        # Update discriminators
        disc_gradients = tape.gradient(
            disc_loss,
            self.model.disc_A.trainable_variables + self.model.disc_B.trainable_variables
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients,
                self.model.disc_A.trainable_variables + self.model.disc_B.trainable_variables)
        )

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}


def create_trainer(
        model_type: str,
        model: tf.keras.Model,
        config: TrainingConfig,
        experiment_name: str,
        checkpoint_dir: Path
) -> MaterialsTrainer:
    """Create appropriate trainer based on model type."""
    if model_type == 'vae':
        return VAETrainer(model, config, experiment_name, checkpoint_dir)
    elif model_type == 'cyclegan':
        return CycleGANTrainer(model, config, experiment_name, checkpoint_dir)
    else:
        return MaterialsTrainer(model, config, experiment_name, checkpoint_dir)


class DataModule:
    """Data handling for material property datasets."""

    def __init__(
            self,
            data_dir: Path = Path('/data/plastic_grid_road'),
            batch_size: int = 32
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self):
        """Prepare datasets."""
        self.train_data = self._load_dataset('train')
        self.val_data = self._load_dataset('val')
        self.test_data = self._load_dataset('test')

    def _load_dataset(self, split: str) -> tf.data.Dataset:
        """Load and preprocess dataset split."""
        data_path = self.data_dir / f'{split}.h5'
        with h5py.File(data_path, 'r') as f:
            data = {key: f[key][()] for key in f.keys()}

        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(1000).batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Setup data
    data_module = DataModule(
        batch_size=config['batch_size']
    )
    data_module.setup()

    # Create model and trainer
    model = create_model(config['model'])
    trainer = create_trainer(
        model_type=config['model']['type'],
        model=model,
        config=TrainingConfig(**config['training']),
        experiment_name=args.experiment,
        checkpoint_dir=Path(config['checkpoint_dir'])
    )

    # Train model
    trainer.train(data_module.train_data, data_module.val_data)