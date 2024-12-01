from typing import Dict, Optional
import logging
from pathlib import Path


def setup_logging(log_dir: Optional[str] = None) -> None:
    """Configure project-wide logging."""
    log_dir = Path(log_dir or 'logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'project.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict:
    """Load project configuration."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


__version__ = '1.0.0'