# OK_Computer: Advanced Materials Discovery Framework

## Overview
OK_Computer is a comprehensive deep learning framework that combines multiple neural network architectures for rapid materials innovation and discovery. It integrates advanced models like CGCNN, GAT-GNN, MEGNet, SchNet, MPNN, and others with sophisticated analysis tools for quantum computing, energy systems, and chemical analysis.

## Core Dependencies

### Required Libraries
```bash
tensorflow==2.9.0
torch==1.13.0
pytorch-geometric==2.2.0
ray[tune]==2.2.0
numpy>=1.21.0
pyyaml>=6.0
ase>=3.22.1
scikit-learn>=1.0.2
pymatgen>=2022.0.0
pandas>=1.4.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.3.0
nglview>=3.0.0
boto3>=1.26.0
```

### Optional Dependencies
```bash
# For quantum computing modules
qiskit>=0.34.0
cirq>=0.13.0

# For advanced visualization
vtk>=9.1.0
mayavi>=4.7.0
```

## Installation

### Method 1: Conda Installation
```bash
# Create conda environment
conda create -n ok_computer python=3.9
conda activate ok_computer

# Install CUDA if using GPU
conda install cudatoolkit=11.3 cudnn=8.2.1

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Pip Installation with Virtual Environment
```bash
# Create virtual environment
python -m venv ok_computer_env
source ok_computer_env/bin/activate  # Linux/Mac
.\ok_computer_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install CUDA support
pip install tensorflow[cuda]
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### AWS Configuration
```bash
# Configure AWS credentials
aws configure
# Enter AWS credentials when prompted
```

## Project Structure
```
ok_computer/
├── config/
│   ├── config.yaml           # Main configuration
│   ├── models/              # Model-specific configs
│   └── applications/        # Application-specific configs
├── models/
│   ├── base_models/         # Core implementations
│   │   ├── cgcnn.py
│   │   ├── gatgnn.py
│   │   ├── megnet.py
│   │   ├── schnet.py
│   │   └── gcn.py
│   ├── enhanced_models/     # Advanced implementations
│   │   ├── super_cgcnn.py
│   │   ├── super_megnet.py
│   │   ├── super_mpnn.py
│   │   └── super_schnet.py
│   └── specialized/         # Domain-specific models
│       ├── quantum_gnn.py
│       ├── energy_net.py
│       └── chemical_vae.py
├── notebooks/
│   ├── quantum/
│   │   ├── quantum_computing_materials_explorer.ipynb
│   │   └── quantum_properties_analyzer.ipynb
│   ├── energy/
│   │   ├── advanced_energy_photovoltaics.ipynb
│   │   └── energy_conversion_analysis.ipynb
│   ├── chemical/
│   │   ├── chemical_system_explorer.ipynb
│   │   └── reaction_pathway_analyzer.ipynb
│   └── analysis/
│       ├── material_discovery.ipynb
│       └── energy_decomposition.ipynb
├── data/
│   ├── raw/                # Raw crystal data
│   ├── processed/          # Processed graph data
│   └── results/           # Analysis results
└── utils/
    ├── preprocessing/     # Data processing utilities
    ├── analysis/         # Analysis tools
    └── visualization/    # Visualization utilities
```

## Configuration

### Main Configuration (config.yaml)
```yaml
data:
  s3:
    bucket: new-materials
    training_prefix: training
    region: us-east-1
  paths:
    base_paths:
      - "ok_computer/data"
      - "New_Materials_Discovery/data"
    subdirs:
      train: "train"
      validation: "validation"
      test: "test"

models:
  super_megnet:
    hidden_dim: 64
    num_blocks: 3
    activation: "softplus"
    use_group_norm: true

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 1e-3
  
analysis:
  quantum_gan:
    num_qubits: 5
    num_shots: 1000
  air_stability:
    include_water: true
    include_co2: true
```

## Running the Framework

### Single Model Execution
```bash
# Run specific model
python -m ok_computer.models.base_models.cgcnn --config config/config.yaml

# Run with custom parameters
python -m ok_computer.models.enhanced_models.super_schnet \
    --config config/config.yaml \
    --batch_size 64 \
    --learning_rate 0.001
```

### Multiple Model Execution (main.py)
```bash
# Run all models
python main.py --all --config config/config.yaml

# Run specific models in parallel
python main.py \
    --models cgcnn schnet megnet \
    --config config/config.yaml \
    --parallel \
    --gpu 0,1

# Run with hyperparameter optimization
python main.py \
    --models super_megnet \
    --config config/config.yaml \
    --optimize \
    --trials 50
```

Command-line options:
- `--models`: Specify models to run
- `--parallel`: Enable parallel execution
- `--gpu`: Specify GPU devices
- `--batch_size`: Override batch size
- `--optimize`: Enable hyperparameter optimization
- `--trials`: Number of optimization trials
- `--all`: Run all available models
- `--config`: Path to configuration file

## Jupyter Notebook Usage

### Starting Jupyter
```bash
jupyter notebook
```

### Available Notebooks

#### 1. Quantum Computing Materials Explorer
**Location**: `notebooks/quantum/quantum_computing_materials_explorer.ipynb`
- Advanced quantum material analysis
- Coherence time optimization
- Multi-qubit coupling analysis
- Quantum noise modeling

```python
# Example usage
from ok_computer.quantum import QuantumMaterialsExplorer

# Initialize with custom configuration
config = QuantumMaterialConfig(
    temperature_range=(0.01, 4.0),
    frequency_range=(1e9, 10e9),
    magnetic_field_range=(0, 1.0),
    coupling_strength_min=10e6,
    decoherence_time_min=1e-6,
    gate_fidelity_min=0.99
)

explorer = QuantumMaterialsExplorer(config)

# Analyze coherence
coherence_results = explorer.analyze_coherence(
    "Al",
    temperature=0.02,
    magnetic_field=0.0
)

# Optimize qubit design
design = explorer.optimize_qubit_design(
    target_frequency=5e9,
    target_coupling=100e6,
    max_crosstalk=-30
)
```

#### 2. Advanced Energy Photovoltaics
**Location**: `notebooks/energy/advanced_energy_photovoltaics.ipynb`
- Deep learning for photovoltaic materials
- Band gap analysis
- Carrier transport optimization
- Device structure optimization

```python
# Example usage
from ok_computer.energy import DeviceOptimizer

config = AdvancedMaterialsConfig(
    temperature=300.0,
    electric_field=1e4,
    carrier_density=1e16,
    defect_density=1e14,
    bandgap_range=(0.9, 2.1)
)

optimizer = DeviceOptimizer(config, target_efficiency=30.0)

# Optimize heterojunction
interface, properties = optimizer.optimize_heterojunction(
    material_1,
    material_2,
    n_iterations=1000
)
```

#### 3. Chemical Systems Explorer
**Location**: `notebooks/chemical/chemical_system_explorer.ipynb`
- Chemical stability analysis
- Phase diagram visualization
- Reaction pathway prediction

```python
# Example usage
from ok_computer.chemical import ChemicalSystemAnalyzer

analyzer = ChemicalSystemAnalyzer()

# Analyze chemical system
results = analyzer.analyze_chemical_system(
    chemsys=["Na", "Cl"],
    grouped_entries=entries,
    minimal_entries=data
)
```

#### 4. Energy Decomposition Analysis
**Location**: `notebooks/analysis/energy_decomposition.ipynb`
- Decomposition energy calculations
- Thermodynamic stability analysis
- Competing phase identification

```python
# Example usage
from ok_computer.analysis import DecompositionEnergyAnalyzer

analyzer = DecompositionEnergyAnalyzer()

# Analyze composition
e_decomp, products = analyzer.compute_decomposition_energy(
    composition="Fe2O3",
    energy=-100.0
)

# Analyze competing phases
competing = analyzer.analyze_competing_phases(
    composition="Fe2O3",
    energy=-100.0,
    energy_window=0.1
)
```

#### 5. Materials Discovery Framework
**Location**: `notebooks/analysis/materials_discovery.ipynb`
- Unified materials analysis interface
- Property prediction
- Sustainability analysis

```python
# Example usage
from ok_computer.discovery import MaterialsExplorer

config = MaterialsConfig(
    application="quantum_computing",
    temperature_range=(0.01, 4),
    pressure_range=(1e-6, 1e5),
    stability_threshold=1.0,
    cost_limit=10000,
    environmental_impact=100
)

explorer = MaterialsExplorer(config)

# Screen materials
candidates = explorer.screen_materials(
    target_property="coherence_time",
    minimum_value=100e-6
)

# Analyze sustainability
impact = explorer.analyze_sustainability(composition="Fe2O3")
```

## Model Architecture Details

### Base Models
1. **CGCNN (Crystal Graph CNN)**
   - Crystal structure encoding
   - Graph convolution layers
   - Property prediction

2. **GAT-GNN (Graph Attention Networks)**
   - Attention mechanisms
   - Multi-head attention
   - Edge feature processing

3. **MEGNet (Materials Energy Graph Network)**
   - Energy-based representation
   - Multi-task learning
   - State tracking

4. **SchNet**
   - Continuous filter convolutions
   - Interaction blocks
   - Energy prediction

### Enhanced Models
1. **Super CGCNN**
   - Advanced pooling mechanisms
   - Residual connections
   - Batch normalization

2. **Super MEGNet**
   - Enhanced energy representation
   - Improved state updates
   - Multi-scale features

3. **Super MPNN**
   - Advanced message passing
   - Global attention
   - Edge updates

4. **Super SchNet**
   - Extended interaction range
   - Improved filter networks
   - Enhanced energy prediction

## Data Processing

### Loading from S3
```python
import boto3
from ok_computer.utils import S3DataLoader

s3_loader = S3DataLoader(
    bucket="new-materials",
    prefix="training"
)

# Load datasets
train_data = s3_loader.load_dataset("train")
val_data = s3_loader.load_dataset("validation")
test_data = s3_loader.load_dataset("test")
```

### Data Pipeline
```python
from ok_computer.pipelines import MaterialsPipeline

pipeline = MaterialsPipeline(
    config_path="config/config.yaml",
    model_type="super_megnet",
    data_source="s3"
)

# Process and train
history = pipeline.train(train_data, val_data)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions
- Include unit tests

## Citation
```bibtex
@article{cette2024okcomputer,
    title={OK_Computer: A Deep Learning Framework for Materials Discovery},
    author={Cette AI},
    year={2024}
}
```

## License
Apache License 2.0

## Contact
michael.lafave@cette.ai

---
Made with ♥️ by Cette AI
