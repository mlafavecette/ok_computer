"""
setup_data.py - Materials Dataset Setup Script
Copyright 2024 Cette AI
Licensed under the Apache License, Version 2.0

Script to download and prepare datasets for materials engineering model training.
Place this file in the /scripts directory of your project.

Usage:
    python setup_data.py [options]

Example:
    python setup_data.py --data_dir ../data --datasets structures compositions
    python setup_data.py --download_all --use_cache
    python setup_data.py --validate_only

    # From project root:
python scripts/setup_data.py --download_all

# Or from scripts directory:
cd scripts
python setup_data.py --download_all

# Download specific datasets:
python setup_data.py --datasets structures compositions

# Download with caching for faster training:
python setup_data.py --download_all --use_cache --batch_size 64

# Validate existing datasets:
python setup_data.py --validate_only

Author: Michael R. Lafave
Last Modified: 2022-11-19
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from materials_ml.data import MaterialsDataManager, DataConfig


def setup_logging():
    """Configure logging output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_setup.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Setup materials science datasets for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all datasets to default location
    python setup_data.py --download_all

    # Download specific datasets to custom location
    python setup_data.py --data_dir /path/to/data --datasets structures compositions

    # Validate existing datasets
    python setup_data.py --validate_only

    # Download and preprocess with caching
    python setup_data.py --download_all --use_cache --batch_size 64
    """
    )

    parser.add_argument(
        "--data_dir",
        default=str(project_root / "data"),
        help="Directory to store datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["structures", "compositions", "summary", "formulas"],
        help="Specific datasets to download"
    )
    parser.add_argument(
        "--download_all",
        action="store_true",
        help="Download all available datasets"
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate existing datasets"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Enable dataset caching"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for preprocessing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload of existing files"
    )

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Configure data manager
        config = DataConfig(
            data_dir=args.data_dir,
            cache_dir=str(Path(args.data_dir) / "cache"),
            batch_size=args.batch_size
        )
        manager = MaterialsDataManager(config)

        if args.validate_only:
            logger.info("Validating existing datasets...")
            manager.validate_datasets()
            return

        # Determine datasets to download
        datasets = None if args.download_all else args.datasets

        # Download and process datasets
        manager.download_datasets(
            datasets,
            force=args.force
        )

        # Validate downloads
        logger.info("Validating downloads...")
        manager.validate_datasets()

        # Process if caching requested
        if args.use_cache:
            logger.info("Processing datasets for caching...")
            manager.prepare_cache(datasets)

        logger.info("Setup complete! Data ready for training.")

    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        raise


if __name__ == "__main__":
    main()