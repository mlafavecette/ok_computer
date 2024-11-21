from .process import (
    MaterialsDataset,
    split_dataset,
    split_dataset_cv,
    create_global_features,
    process_crystal_data,
    DataProcessor,
    SOAPFeaturizer,
    SMFeaturizer,
    GaussianDistance,
    AtomFeaturizer,
    VoronoiFeaturizer,
    EdgeFeaturizer
)

__version__ = '2.0.0'
__author__ = 'Michael Lafave'

# Export key configurations
DEFAULT_PROCESSING_CONFIG = {
    'max_neighbors': 12,
    'max_radius': 8.0,
    'edge_features': True,
    'node_features': True,
    'global_features': True,
    'use_voronoi': False,
    'use_soap': False,
    'use_sm': False,
    'gaussian_distance_width': 0.2,
    'gaussian_distance_bins': 50,
    'batch_size': 32,
    'cache_data': True,
    'num_parallel_calls': 8
}


# Initialize optional dependencies
def check_optional_dependencies():
    """Check and report status of optional dependencies."""
    optional_deps = {
        'pymatgen': 'For Voronoi features',
        'dscribe': 'For SOAP/SM descriptors',
        'ase': 'For structure handling',
        'keras': 'For neural network models'
    }

    missing = []
    for package, purpose in optional_deps.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(f"{package} ({purpose})")

    if missing:
        print("Optional dependencies not found:")
        for pkg in missing:
            print(f"- {pkg}")

    return len(missing) == 0


# Automatically check dependencies on import
has_all_deps = check_optional_dependencies()
