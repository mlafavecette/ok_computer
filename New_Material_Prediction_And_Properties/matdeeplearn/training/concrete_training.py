"""
from training.concrete_training import main
main()
"""

import sys
from pathlib import Path
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from typing import Tuple, Dict

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models.vae_concrete import ConcreteVAE
from process.process import ConcreteDataProcessor


class ConcreteVAETrainer:
    """Handles training of the concrete VAE model."""

    def __init__(
            self,
            model: ConcreteVAE,
            data_processor: ConcreteDataProcessor,
            config: Dict = None
    ):
        self.model = model
        self.data_processor = data_processor
        self.config = config or self._default_config()

        # Setup paths
        self.save_dir = os.path.join(project_root, "models", "saved")
        self.log_dir = os.path.join(project_root, "logs")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _default_config(self) -> Dict:
        """Default training configuration."""
        return {
            "batch_size": 64,
            "epochs": 200,
            "learning_rate": 1e-4,
            "validation_split": 0.2,
            "early_stopping_patience": 20
        }

    def prepare_callbacks(self) -> List:
        """Setup training callbacks."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.save_dir, f"vae_checkpoint_{timestamp}"),
                save_best_only=True,
                monitor="val_loss"
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.log_dir, f"run_{timestamp}"),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config["early_stopping_patience"],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> tf.keras.callbacks.History:
        """Train the VAE model."""
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config["learning_rate"])
        )

        # Train model
        history = self.model.fit(
            [X_train, y_train],
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_split=self.config["validation_split"],
            callbacks=self.prepare_callbacks(),
            verbose=1
        )

        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model.save(os.path.join(self.save_dir, f"concrete_vae_final_{timestamp}"))

        return history

    def evaluate_model(
            self,
            X_test: np.ndarray,
            y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        results = self.model.evaluate([X_test, y_test], verbose=0)

        metrics = {
            "total_loss": results[0],
            "reconstruction_loss": results[1],
            "kl_loss": results[2],
            "strength_loss": results[3],
            "carbon_loss": results[4]
        }

        return metrics


def main():
    # Initialize components
    data_processor = ConcreteDataProcessor()
    X_train, y_train = data_processor.generate_training_data()

    model = ConcreteVAE(
        original_dim=21,  # Number of concrete components
        latent_dim=32,
        encoder_hidden_dims=[256, 128, 64],
        decoder_hidden_dims=[64, 128, 256],
        beta=1.0
    )

    # Create trainer and train model
    trainer = ConcreteVAETrainer(model, data_processor)
    history = trainer.train(X_train, y_train)

    # Generate example optimization
    target_strength = (50.0, 5.0, 3.5)  # MPa
    target_carbon = (-326.45, 435.0)  # kg CO2/tonne

    optimal_mixture = model.generate_optimized_mixture(
        target_strength,
        target_carbon
    )

    # Print results
    component_names = list(data_processor.base_composition.keys())
    print("\nOptimized Concrete Mixture:")
    for i, component in enumerate(component_names):
        print(f"{component}: {optimal_mixture[i]:.2f}%")


if __name__ == "__main__":
    main()