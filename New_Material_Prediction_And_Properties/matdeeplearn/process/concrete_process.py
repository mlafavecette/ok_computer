"""
/process/process.py
Data processing module for concrete composition analysis.
"""

import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ConcreteDataProcessor:
    """Processes and generates concrete composition data."""

    def __init__(self):
        # Initialize base composition from document
        self.base_composition = {
            "Fly Ash (Class F)": {"percentage": 30, "unit_cost": 0.02},
            "GGBFS": {"percentage": 20, "unit_cost": 0.05},
            "Metakaolin": {"percentage": 5, "unit_cost": 0.20},
            "Silica Fume": {"percentage": 3, "unit_cost": 0.30},
            "Magnesium Silicate": {"percentage": 15, "unit_cost": 0.10},
            "Olivine": {"percentage": 15, "unit_cost": 0.05},
            "Biochar": {"percentage": 10, "unit_cost": 0.50},
            "Basalt Rock Dust": {"percentage": 2, "unit_cost": 0.02},
            "Calcium Carbonate": {"percentage": 3, "unit_cost": 0.05},
            "Magnesium Carbonate": {"percentage": 2, "unit_cost": 0.05},
            "Natural Aggregates": {"percentage": 10, "unit_cost": 0.02},
            "Recycled Aggregates": {"percentage": 5, "unit_cost": 0.01},
            "Sodium Hydroxide": {"percentage": 2, "unit_cost": 0.50},
            "Sodium Silicate": {"percentage": 4, "unit_cost": 0.40},
            "Silica from Algae": {"percentage": 1, "unit_cost": 1.00},
            "Algae Biomass": {"percentage": 3, "unit_cost": 1.00},
            "Alginate Beads": {"percentage": 5, "unit_cost": 1.00},
            "Carbonation Accelerators": {"percentage": 2, "unit_cost": 0.50},
            "Water": {"percentage": 5, "unit_cost": 0.001},
            "Superplasticizers": {"percentage": 0.5, "unit_cost": 1.50},
            "Air-Entraining Agents": {"percentage": 0.1, "unit_cost": 1.50}
        }

        # Environmental properties
        self.emission_factors = {
            "Fly Ash (Class F)": 0.01,
            "GGBFS": 0.05,
            "Metakaolin": 0.15,
            "Silica Fume": 0.02,
            "Magnesium Silicate": 0.05,
            "Olivine": 0.05,
            "Biochar": 0.00,
            "Basalt Rock Dust": 0.01,
            "Calcium Carbonate": 0.05,
            "Magnesium Carbonate": 0.05,
            "Natural Aggregates": 0.01,
            "Recycled Aggregates": 0.005,
            "Sodium Hydroxide": 1.00,
            "Sodium Silicate": 0.80,
            "Silica from Algae": 0.00,
            "Algae Biomass": 0.00,
            "Alginate Beads": 0.00,
            "Carbonation Accelerators": 0.50,
            "Water": 0.00,
            "Superplasticizers": 1.00,
            "Air-Entraining Agents": 1.50
        }

        self.sequestration_rates = {
            "Magnesium Silicate": 0.30,
            "Olivine": 0.30,
            "Biochar": 2.00,
            "Algae Biomass": 1.00,
            "Alginate Beads": 1.00,
            "Calcium Carbonate": 0.10,
            "Magnesium Carbonate": 0.10,
            "Carbonation Accelerators": 0.50
        }

        # Initialize scalers
        self.composition_scaler = StandardScaler()
        self.property_scaler = StandardScaler()

    def generate_training_data(
            self,
            n_samples: int = 10000,
            variation_range: float = 0.2,
            random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for concrete VAE.

        Args:
            n_samples: Number of synthetic samples
            variation_range: Range of random variation
            random_state: Random seed for reproducibility

        Returns:
            Normalized composition and property arrays
        """
        np.random.seed(random_state)

        compositions = []
        properties = []

        for _ in range(n_samples):
            # Generate composition
            composition = self._generate_composition_variation(variation_range)
            compositions.append(composition)

            # Calculate properties
            mechanical_props = self._calculate_mechanical_properties(composition)
            carbon_props = self._calculate_carbon_metrics(composition)

            properties.append(np.concatenate([mechanical_props, carbon_props]))

        # Convert to arrays
        X = np.array(compositions)
        y = np.array(properties)

        # Normalize data
        X_normalized = self.composition_scaler.fit_transform(X)
        y_normalized = self.property_scaler.fit_transform(y)

        return X_normalized, y_normalized

    def _generate_composition_variation(self, variation_range: float) -> np.ndarray:
        """Generate valid composition variation."""
        composition = []
        total = 0

        # Generate variations
        for material in self.base_composition:
            base_pct = self.base_composition[material]["percentage"]
            variation = np.random.uniform(
                -variation_range * base_pct,
                variation_range * base_pct
            )
            adjusted_pct = max(0, base_pct + variation)
            composition.append(adjusted_pct)
            total += adjusted_pct

        # Normalize to 100%
        composition = np.array(composition) * (100 / total)

        return composition

    def _calculate_mechanical_properties(self, composition: np.ndarray) -> np.ndarray:
        """Calculate concrete mechanical properties."""
        # Base properties from the optimal mixture
        base_properties = {
            "compression": 50.0,  # MPa
            "flexural": 5.0,  # MPa
            "tensile": 3.5  # MPa
        }

        # Material contribution factors
        strength_factors = {
            "GGBFS": 1.2,
            "Silica Fume": 1.3,
            "Metakaolin": 1.1,
            "Fly Ash (Class F)": 0.9,
            "Natural Aggregates": 1.0
        }

        # Calculate property modifications
        property_mods = {prop: 1.0 for prop in base_properties}

        for i, (material, _) in enumerate(self.base_composition.items()):
            if material in strength_factors:
                factor = strength_factors[material]
                ratio = composition[i] / self.base_composition[material]["percentage"]
                for prop in property_mods:
                    property_mods[prop] *= (ratio * factor) ** 0.5

        return np.array([
            base_properties["compression"] * property_mods["compression"],
            base_properties["flexural"] * property_mods["flexural"],
            base_properties["tensile"] * property_mods["tensile"]
        ])

    def _calculate_carbon_metrics(self, composition: np.ndarray) -> np.ndarray:
        """Calculate CO2 emissions and sequestration."""
        # Calculate emissions
        emissions = sum(
            composition[i] * self.emission_factors[material]
            for i, material in enumerate(self.base_composition)
        )

        # Calculate sequestration
        sequestration = sum(
            composition[i] * self.sequestration_rates[material]
            for i, material in enumerate(self.base_composition)
            if material in self.sequestration_rates
        )

        # Add CO2 curing process contribution
        sequestration += 50  # per 1000 tonnes

        return np.array([emissions, sequestration])

    def denormalize_composition(self, normalized_composition: np.ndarray) -> np.ndarray:
        """Convert normalized composition back to original scale."""
        return self.composition_scaler.inverse_transform(normalized_composition)

    def denormalize_properties(self, normalized_properties: np.ndarray) -> np.ndarray:
        """Convert normalized properties back to original scale."""
        return self.property_scaler.inverse_transform(normalized_properties)

    def validate_composition(self, composition: np.ndarray) -> bool:
        """Validate if a composition is physically realistic."""
        if not (99.9 <= np.sum(composition) <= 100.1):
            return False

        # Check component ranges
        for i, (material, props) in enumerate(self.base_composition.items()):
            base_pct = props["percentage"]
            if composition[i] < 0 or composition[i] > base_pct * 2:
                return False

        return True

    def calculate_cost(self, composition: np.ndarray) -> float:
        """Calculate cost per tonne of concrete mixture."""
        return sum(
            composition[i] * self.base_composition[material]["unit_cost"]
            for i, material in enumerate(self.base_composition)
        )


if __name__ == "__main__":
    # Example usage
    processor = ConcreteDataProcessor()
    X, y = processor.generate_training_data(n_samples=1000)

    print("Generated training data shapes:")
    print(f"Compositions: {X.shape}")
    print(f"Properties: {y.shape}")

    # Example composition validation
    example_comp = processor.denormalize_composition(X[0].reshape(1, -1))[0]
    print("\nExample composition validation:", processor.validate_composition(example_comp))
    print("Example composition cost: $", processor.calculate_cost(example_comp))