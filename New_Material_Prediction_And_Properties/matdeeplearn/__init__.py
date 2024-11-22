"""
New_Material_Prediction_And_Properties
====================================

A comprehensive package for material property prediction and process optimization,
specializing in plastic-to-fuel conversion using advanced neural network architectures.

Main Components:
---------------
- models: Neural network architectures for material prediction
- training: Training utilities and data management
- process: Process optimization and monitoring tools
# Import the package
import New_Material_Prediction_And_Properties as nmp

# Get available models
models = nmp.get_available_models()

# Create a specific model
schnet = nmp.get_model('SchNet')

# Use training utilities
trainer = nmp.Trainer()
"""

import os
import sys
import logging
from importlib import import_module
from pathlib import Path

# Package metadata
__version__ = '1.0.0'
__author__ = 'OK Computer Team'
__license__ = 'Cette'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add package root to Python path
package_root = Path(__file__).parent
sys.path.append(str(package_root))

# Import submodules
try:
    # Models
    from .models import (
        SchNet,
        MPNN,
        MEGNet,
        CGCNN,
        GAT_GNN,
        DeepGATGNN,
        GCN,
        SuperSchNet,
        DescriptorNN,
        SuperMPNN,
        SuperMEGNet,
        SuperCGCNN,
        QuantumGAN
    )

    # Training utilities
    from .training import (
        Trainer,
        DataLoader,
        LossFunction,
        Metrics,
        Validator
    )

    # Process optimization
    from .process import (
        Optimizer,
        Monitor,
        Controller,
        Predictor
    )

except ImportError as e:
    logger.warning(f"Some modules could not be imported: {str(e)}")

# Define package exports
__all__ = [
    # Models
    'SchNet',
    'MPNN',
    'MEGNet',
    'CGCNN',
    'GAT_GNN',
    'DeepGATGNN',
    'GCN',
    'SuperSchNet',
    'DescriptorNN',
    'SuperMPNN',
    'SuperMEGNet',
    'SuperCGCNN',
    'QuantumGAN',

    # Training
    'Trainer',
    'DataLoader',
    'LossFunction',
    'Metrics',
    'Validator',

    # Process
    'Optimizer',
    'Monitor',
    'Controller',
    'Predictor'
]


def get_version():
    """Return the package version."""
    return __version__


def get_available_models():
    """Return a list of available model architectures."""
    return [
        'SchNet',
        'MPNN',
        'MEGNet',
        'CGCNN',
        'GAT_GNN',
        'DeepGATGNN',
        'GCN',
        'SuperSchNet',
        'DescriptorNN',
        'SuperMPNN',
        'SuperMEGNet',
        'SuperCGCNN',
        'QuantumGAN'
    ]


def get_model(name: str):
    """
    Get a specific model by name.

    Args:
        name (str): Name of the model to retrieve

    Returns:
        Model class if found, else raises ValueError
    """
    if name not in get_available_models():
        raise ValueError(f"Model {name} not found. Available models: {get_available_models()}")

    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError:
        raise ImportError(f"Model {name} is listed but not properly imported")


# Package initialization checks
def _check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'tensorflow',
        'torch',
        'numpy',
        'pandas',
        'scipy',
        'sklearn'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            import_module(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.warning("Some functionality may be limited")


# Run initialization checks
_check_dependencies()