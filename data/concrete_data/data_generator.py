import numpy as np
import pandas as pd
from scipy.stats import norm, uniform
import json
from pathlib import Path
import random
from datetime import datetime, timedelta


class ConcreteDataGenerator:
    def __init__(self, base_path="concrete_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Define baseline properties for Portland cement
        self.portland_baseline = {
            'compression_strength': 40.0,  # MPa
            'tensile_strength': 3.5,  # MPa
            'flexural_strength': 4.5,  # MPa
            'co2_emissions': 800.0,  # kg/tonne
            'co2_capture': 0.0,  # kg/tonne
            'mineralization': 0.0,  # kg/tonne
            'direct_air_capture': 0.0  # kg/tonne
        }

        # Define composition 1 (from first document)
        self.composition1 = {
            'name': 'MinusZero_Composition1',
            'compression_strength': 42.0,  # MPa
            'tensile_strength': 3.8,  # MPa
            'flexural_strength': 4.8,  # MPa
            'co2_emissions': 64.2,  # kg/tonne
            'co2_capture': 101.0,  # kg/tonne
            'mineralization': 70.0,  # kg/tonne
            'direct_air_capture': 31.0,  # kg/tonne
            'components': {
                'fly_ash': 0.22,
                'ggbfs': 0.20,
                'metakaolin': 0.04,
                'recycled_aggregates': 0.20,
                'natural_aggregates': 0.12,
                'silica_fume': 0.03,
                'sodium_hydroxide': 0.01,
                'sodium_silicate': 0.02,
                'algae_biomass': 0.03,
                'calcium_carbonate': 0.03,
                'magnesium_carbonate': 0.03,
                'silica_from_algae': 0.01,
                'magnesium_silicate': 0.02,
                'olivine': 0.03,
                'basalt_rock_dust': 0.02,
                'biochar': 0.02,
                'alginate_beads': 0.03,
                'carbonation_accelerators': 0.02,
                'water': 0.01
            }
        }

        # Define composition 2 (from second document)
        self.composition2 = {
            'name': 'MinusZero_Composition2',
            'compression_strength': 45.0,  # MPa
            'tensile_strength': 4.0,  # MPa
            'flexural_strength': 5.0,  # MPa
            'co2_emissions': 108.55,  # kg/tonne
            'co2_capture': 435.0,  # kg/tonne
            'mineralization': 155.0,  # kg/tonne
            'direct_air_capture': 280.0,  # kg/tonne
            'components': {
                'fly_ash': 0.30,
                'ggbfs': 0.20,
                'metakaolin': 0.05,
                'silica_fume': 0.03,
                'magnesium_silicate': 0.15,
                'olivine': 0.15,
                'biochar': 0.10,
                'basalt_rock_dust': 0.02,
                'calcium_carbonate': 0.03,
                'magnesium_carbonate': 0.02,
                'natural_aggregates': 0.10,
                'recycled_aggregates': 0.05,
                'sodium_hydroxide': 0.02,
                'sodium_silicate': 0.04,
                'silica_from_algae': 0.01,
                'algae_biomass': 0.03,
                'alginate_beads': 0.05,
                'carbonation_accelerators': 0.02,
                'water': 0.05
            }
        }

    def generate_variation(self, base_composition, num_samples):
        variations = []

        for _ in range(num_samples):
            variation = {
                'name': base_composition['name'],
                'timestamp': datetime.now() + timedelta(minutes=random.randint(0, 1000000)),
                'temperature': random.uniform(15, 30),  # Curing temperature
                'humidity': random.uniform(40, 90),  # Curing humidity
                'age_days': random.randint(7, 365)  # Concrete age
            }

            # Add strength variations with realistic correlations
            base_compression = base_composition['compression_strength']
            variation['compression_strength'] = max(0, np.random.normal(
                base_compression,
                base_compression * 0.1
            ))

            # Correlate tensile and flexural strength with compression strength
            strength_factor = variation['compression_strength'] / base_compression
            variation['tensile_strength'] = max(0, np.random.normal(
                base_composition['tensile_strength'] * strength_factor,
                base_composition['tensile_strength'] * 0.05
            ))
            variation['flexural_strength'] = max(0, np.random.normal(
                base_composition['flexural_strength'] * strength_factor,
                base_composition['flexural_strength'] * 0.05
            ))

            # Add CO2-related variations
            variation['co2_emissions'] = max(0, np.random.normal(
                base_composition['co2_emissions'],
                base_composition['co2_emissions'] * 0.05
            ))
            variation['co2_capture'] = max(0, np.random.normal(
                base_composition['co2_capture'],
                base_composition['co2_capture'] * 0.05
            ))
            variation['mineralization'] = max(0, np.random.normal(
                base_composition['mineralization'],
                base_composition['mineralization'] * 0.05
            ))
            variation['direct_air_capture'] = max(0, np.random.normal(
                base_composition['direct_air_capture'],
                base_composition['direct_air_capture'] * 0.05
            ))

            # Add component variations
            variation['components'] = {}
            total_adjustment = 0
            for component, ratio in base_composition['components'].items():
                # Generate variation while maintaining reasonable bounds
                adjustment = np.random.normal(0, 0.01)
                total_adjustment += adjustment
                variation['components'][component] = max(0, min(1, ratio + adjustment))

            # Normalize components to ensure they sum to 1
            total = sum(variation['components'].values())
            for component in variation['components']:
                variation['components'][component] /= total

            variations.append(variation)

        return variations

    def generate_dataset(self, samples_per_composition=50000):
        """Generate complete dataset with all compositions"""
        all_data = []

        # Generate Portland cement data
        portland_composition = {
            'name': 'Portland_Cement',
            'compression_strength': self.portland_baseline['compression_strength'],
            'tensile_strength': self.portland_baseline['tensile_strength'],
            'flexural_strength': self.portland_baseline['flexural_strength'],
            'co2_emissions': self.portland_baseline['co2_emissions'],
            'co2_capture': self.portland_baseline['co2_capture'],
            'mineralization': self.portland_baseline['mineralization'],
            'direct_air_capture': self.portland_baseline['direct_air_capture'],
            'components': {
                'portland_cement': 0.15,
                'aggregates': 0.75,
                'water': 0.10
            }
        }

        # Generate variations for each composition
        all_data.extend(self.generate_variation(portland_composition, samples_per_composition))
        all_data.extend(self.generate_variation(self.composition1, samples_per_composition))
        all_data.extend(self.generate_variation(self.composition2, samples_per_composition))

        return all_data

    def save_dataset(self, data, format='json'):
        """Save dataset in specified format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format == 'json':
            output_file = self.base_path / f'concrete_data_{timestamp}.json'
            with open(output_file, 'w') as f:
                json.dump(data, f, default=str, indent=2)

        elif format == 'csv':
            # Flatten the data structure for CSV format
            flattened_data = []
            for entry in data:
                flat_entry = {
                    'name': entry['name'],
                    'timestamp': entry['timestamp'],
                    'temperature': entry['temperature'],
                    'humidity': entry['humidity'],
                    'age_days': entry['age_days'],
                    'compression_strength': entry['compression_strength'],
                    'tensile_strength': entry['tensile_strength'],
                    'flexural_strength': entry['flexural_strength'],
                    'co2_emissions': entry['co2_emissions'],
                    'co2_capture': entry['co2_capture'],
                    'mineralization': entry['mineralization'],
                    'direct_air_capture': entry['direct_air_capture']
                }
                # Add component ratios
                for component, value in entry['components'].items():
                    flat_entry[f'component_{component}'] = value
                flattened_data.append(flat_entry)

            df = pd.DataFrame(flattened_data)
            output_file = self.base_path / f'concrete_data_{timestamp}.csv'
            df.to_csv(output_file, index=False)


def main():
    # Initialize generator
    generator = ConcreteDataGenerator()

    # Generate approximately 250MB of data
    # Each sample is roughly 1KB, so we need about 250,000 samples
    samples_per_composition = 83334  # This will give us ~250,000 total samples

    # Generate dataset
    print("Generating dataset...")
    dataset = generator.generate_dataset(samples_per_composition)

    # Save in both formats
    print("Saving dataset...")
    generator.save_dataset(dataset, format='json')
    generator.save_dataset(dataset, format='csv')
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()