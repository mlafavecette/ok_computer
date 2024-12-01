"""
energy_prediction.py - Energy Prediction and Analysis Module
Copyright 2024 Cette
Licensed under the Apache License, Version 2.0

This module implements specialized functionality for energy prediction,
uncertainty estimation, and analysis of crystal structures.

Author: Michael Lafave
Last Modified: 2020-05-20
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


class EnergyPredictor(tf.keras.Model):
    """Energy prediction model with uncertainty estimation.

    This model predicts formation energies and provides uncertainty
    estimates through ensemble or probabilistic predictions.
    """

    def __init__(
            self,
            hidden_dims: int = 128,
            num_layers: int = 4,
            num_ensemble: int = 5,
            dropout_rate: float = 0.1,
            name: str = "energy_predictor"
    ):
        """Initialize the energy predictor.

        Args:
            hidden_dims: Hidden layer dimensions
            num_layers: Number of graph conv layers
            num_ensemble: Number of ensemble members
            dropout_rate: Dropout rate for uncertainty
            name: Model name
        """
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.num_ensemble = num_ensemble
        self.dropout_rate = dropout_rate

        # Build ensemble of models
        self.ensemble = [
            self._build_base_model()
            for _ in range(num_ensemble)
        ]

    def _build_base_model(self) -> tf.keras.Model:
        """Build individual model for ensemble.

        Returns:
            Base model instance
        """
        from .gnome_tf import CrystalGraphNetwork

        return CrystalGraphNetwork(
            num_layers=self.num_layers,
            hidden_dims=self.hidden_dims
        )

    def call(
            self,
            inputs: Dict[str, tf.Tensor],
            training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass with uncertainty estimation.

        Args:
            inputs: Input graph data
            training: Whether in training mode

        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Get predictions from each ensemble member
        predictions = []

        for model in self.ensemble:
            pred = model(inputs, training=training)
            predictions.append(pred)

        # Stack predictions
        predictions = tf.stack(predictions, axis=0)

        # Compute mean and uncertainty
        mean_pred = tf.reduce_mean(predictions, axis=0)
        uncertainty = tf.math.reduce_std(predictions, axis=0)

        return mean_pred, uncertainty

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """Custom training step for ensemble.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dictionary of metrics
        """
        inputs, targets = data

        # Train each ensemble member
        losses = []
        with tf.GradientTape() as tape:
            predictions, uncertainty = self(inputs, training=True)

            # Compute loss with uncertainty regularization
            mse_loss = tf.reduce_mean(tf.square(predictions - targets))
            uncertainty_loss = tf.reduce_mean(
                tf.exp(-uncertainty) * tf.square(predictions - targets)
                + uncertainty
            )

            total_loss = mse_loss + 0.1 * uncertainty_loss

        # Update weights
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)
        )

        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'uncertainty_loss': uncertainty_loss
        }

    def evaluate_structure(
            self,
            structure: Dict[str, tf.Tensor],
            num_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """Detailed evaluation of a crystal structure.

        Args:
            structure: Input crystal structure
            num_samples: Number of Monte Carlo samples

        Returns:
            Dictionary of predictions and analyses
        """
        # Get base prediction and uncertainty
        pred_energy, uncertainty = self(structure, training=False)

        # Monte Carlo dropout sampling
        mc_samples = []
        for _ in range(num_samples):
            pred, _ = self(structure, training=True)  # Enable dropout
            mc_samples.append(pred)

        mc_samples = np.stack(mc_samples)

        return {
            'energy': pred_energy.numpy(),
            'uncertainty': uncertainty.numpy(),
            'mc_mean': np.mean(mc_samples, axis=0),
            'mc_std': np.std(mc_samples, axis=0)
        }


class StabilityAnalyzer:
    """Analyzes crystal structure stability and properties.

    This class provides tools for analyzing the stability of crystal
    structures and predicting their properties.
    """

    def __init__(
            self,
            energy_model: EnergyPredictor,
            temperature: float = 300.0,
            pressure: float = 1.0
    ):
        """Initialize the stability analyzer.

        Args:
            energy_model: Trained energy prediction model
            temperature: Temperature in Kelvin
            pressure: Pressure in atmospheres
        """
        self.energy_model = energy_model
        self.temperature = temperature
        self.pressure = pressure

        # Physical constants
        self.k_B = 8.617333262145e-5  # eV/K
        self.R = 8.31446261815324  # J/mol⋅K

        def analyze_stability(
                self,
                structure: Dict[str, tf.Tensor],
                competing_structures: List[Dict[str, tf.Tensor]] = None
        ) -> Dict[str, float]:
            """Analyze thermodynamic stability of a crystal structure.

            Args:
                structure: Target crystal structure
                competing_structures: List of competing structures

            Returns:
                Dictionary of stability metrics
            """
            # Predict formation energy
            energy, uncertainty = self.energy_model(structure, training=False)

            # Calculate Gibbs free energy
            entropy = self._estimate_vibrational_entropy(structure)
            enthalpy = energy + self.pressure * self._calculate_volume(structure)
            gibbs = enthalpy - self.temperature * entropy

            stability_metrics = {
                'formation_energy': float(energy.numpy()),
                'uncertainty': float(uncertainty.numpy()),
                'gibbs_free_energy': float(gibbs.numpy()),
                'entropy': float(entropy.numpy()),
            }

            if competing_structures:
                # Compute energy above hull
                hull_distance = self._energy_above_hull(
                    structure,
                    competing_structures
                )
                stability_metrics['energy_above_hull'] = float(hull_distance)

                # Estimate decomposition products
                decomp_products = self._predict_decomposition(
                    structure,
                    competing_structures
                )
                stability_metrics['decomposition_products'] = decomp_products

            return stability_metrics

        def _estimate_vibrational_entropy(
                self,
                structure: Dict[str, tf.Tensor]
        ) -> tf.Tensor:
            """Estimate vibrational entropy using Debye model.

            Args:
                structure: Crystal structure

            Returns:
                Estimated vibrational entropy
            """
            # Estimate Debye temperature using force constants
            debye_temp = self._estimate_debye_temperature(structure)

            x = debye_temp / self.temperature

            # Debye model for entropy
            # S = -3R[ln(1-e^(-x)) - (3/4)x - D(x)]
            # where D(x) is the Debye function

            return -3 * self.R * (
                    tf.math.log(1 - tf.exp(-x))
                    - 0.75 * x
                    - self._debye_function(x)
            )

        def _calculate_volume(
                self,
                structure: Dict[str, tf.Tensor]
        ) -> tf.Tensor:
            """Calculate unit cell volume.

            Args:
                structure: Crystal structure

            Returns:
                Unit cell volume
            """
            # Extract lattice vectors
            cell = structure['cell']
            return tf.abs(tf.linalg.det(cell))

        def _energy_above_hull(
                self,
                structure: Dict[str, tf.Tensor],
                competing_structures: List[Dict[str, tf.Tensor]]
        ) -> tf.Tensor:
            """Calculate energy above convex hull.

            Args:
                structure: Target structure
                competing_structures: Competing structures

            Returns:
                Energy above convex hull
            """
            # Get composition and energy for all structures
            compositions = [self._get_composition(s) for s in [structure] + competing_structures]
            energies = [float(self.energy_model(s)[0].numpy()) for s in [structure] + competing_structures]

            # Construct convex hull
            from scipy.spatial import ConvexHull
            points = np.array(list(zip(compositions, energies)))
            hull = ConvexHull(points)

            # Calculate distance to hull
            target_point = points[0]
            hull_points = points[hull.vertices]
            distance = self._point_to_hull_distance(target_point, hull_points)

            return tf.constant(distance, dtype=tf.float32)

        def _predict_decomposition(
                self,
                structure: Dict[str, tf.Tensor],
                competing_structures: List[Dict[str, tf.Tensor]]
        ) -> List[Tuple[str, float]]:
            """Predict decomposition products if unstable.

            Args:
                structure: Target structure
                competing_structures: Competing structures

            Returns:
                List of (formula, fraction) tuples
            """
            # Get hull distance
            hull_distance = self._energy_above_hull(structure, competing_structures)

            if hull_distance < 1e-3:  # Structure is stable
                return []

            # Find closest stable structures on hull
            target_comp = self._get_composition(structure)
            stable_structures = []

            for struct in competing_structures:
                comp = self._get_composition(struct)
                energy = float(self.energy_model(struct)[0].numpy())
                stable_structures.append((comp, energy, struct))

            # Use linear programming to find decomposition
            from scipy.optimize import linprog

            # Setup optimization problem
            result = linprog(
                c=[s[1] for s in stable_structures],
                A_eq=[target_comp],
                b_eq=[1.0],
                bounds=[(0, None) for _ in stable_structures]
            )

            if not result.success:
                warnings.warn("Could not find valid decomposition path")
                return []

            # Get decomposition products
            products = []
            for coeff, (_, _, struct) in zip(result.x, stable_structures):
                if coeff > 1e-3:  # Filter out negligible contributions
                    formula = self._get_formula(struct)
                    products.append((formula, float(coeff)))

            return products

        def screen_stability(
                self,
                structures: List[Dict[str, tf.Tensor]],
                threshold: float = 0.1
        ) -> List[Dict[str, tf.Tensor]]:
            """Screen structures for stability.

            Args:
                structures: List of structures to screen
                threshold: Energy threshold for stability

            Returns:
                List of stable structures
            """
            stable_structures = []

            for struct in structures:
                metrics = self.analyze_stability(struct)

                if metrics['energy_above_hull'] < threshold:
                    stable_structures.append(struct)

            return stable_structures

        def _get_composition(self, structure: Dict[str, tf.Tensor]) -> np.ndarray:
            """Get normalized composition vector."""
            atomic_numbers = structure['nodes']
            unique, counts = np.unique(atomic_numbers, return_counts=True)
            total = np.sum(counts)
            return counts / total

        def _get_formula(self, structure: Dict[str, tf.Tensor]) -> str:
            """Get chemical formula string."""
            atomic_numbers = structure['nodes']
            unique, counts = np.unique(atomic_numbers, return_counts=True)

            # Convert atomic numbers to symbols
            from mendeleev import element
            symbols = [element(int(z)).symbol for z in unique]

            # Construct formula string
            formula_parts = []
            for symbol, count in zip(symbols, counts):
                if count == 1:
                    formula_parts.append(symbol)
                else:
                    formula_parts.append(f"{symbol}{count}")

            return "".join(formula_parts)

        @staticmethod
        def _debye_function(x: tf.Tensor) -> tf.Tensor:
            """Compute Debye function D(x)."""

            def integrand(t):
                return t ** 3 / (tf.exp(t) - 1)

            # Numerical integration
            from scipy.integrate import quad
            result = quad(integrand, 0, float(x))[0]
            return tf.constant(3 / x ** 3 * result, dtype=tf.float32)

        @staticmethod
        def _point_to_hull_distance(
                point: np.ndarray,
                hull_points: np.ndarray
        ) -> float:
            """Calculate shortest distance from point to convex hull."""
            from scipy.spatial.distance import cdist

            # Find nearest point on hull
            distances = cdist([point], hull_points)
            return float(np.min(distances))

        def _estimate_debye_temperature(
                self,
                structure: Dict[str, tf.Tensor]
        ) -> tf.Tensor:
            """Estimate Debye temperature from structure."""
            # This is a simplified estimate
            volume = self._calculate_volume(structure)
            num_atoms = tf.shape(structure['nodes'])[0]

            # Rough estimate based on empirical relationships
            return tf.sqrt(num_atoms / volume) * 100  # Scaling factor
