"""
Configuration management for New_Material_Prediction_And_Properties package.
Handles loading and validation of model and training configurations.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation for models and training."""

    def __init__(self):
        self.config_dir = Path(__file__).parent
        self.model_config = self._load_config('model_config.yaml')
        self.training_config = self._load_config('training_config.yaml')

    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_dir / filename, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file {filename} not found. Using defaults.")
            return {}

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model."""
        return self.model_config.get(model_name, {})

    def get_training_config(self, model_name: str) -> Dict[str, Any]:
        """Get training configuration for specific model."""
        return self.training_config.get(model_name, self.training_config.get('default', {}))

    def validate_config(self, config: Dict[str, Any], config_type: str) -> bool:
        """Validate configuration against schema."""
        # Implementation would include detailed validation logic
        return True


# Initialize global config manager
config_manager = ConfigManager()

# Export config manager instance
__all__ = ['config_manager']