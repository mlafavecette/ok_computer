

```markdown
# OK_Computer Neural Network Framework

## Accelerating Materials Discovery Through Deep Learning

OK_Computer is a comprehensive deep learning framework combining multiple neural network architectures for rapid materials innovation. It integrates CGCNN, GAT-GNN, MEGNet, SchNet, MPNN, GCN, VAE and other networks to enable accelerated materials discovery.

## Prerequisites

### Core Dependencies
```bash
tensorflow==2.9.0
torch==1.13.0
pytorch-geometric==2.2.0
ray[tune]==2.2.0
numpy>=1.21.0
pyyaml>=6.0
ase>=3.22.1
scikit-learn>=1.0.2
```

### Additional Libraries
```bash
pandas>=1.4.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Installation

1. Create and activate conda environment:
```bash
conda create -n ok_computer python=3.9
conda activate ok_computer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
conda install cudatoolkit=11.3  # If using GPU
```

3. Clone repository:
```bash
git clone https://github.com/cette/ok_computer.git
cd ok_computer
```

## Quick Start

### 1. Data Structure
```
data/
├── structures/         # Crystal structures in JSON format
├── targets.csv        # Properties to predict
└── metadata.json      # Optional metadata
```

### 2. Configuration
Create `config.yml`:
```yaml
Job:
  run_mode: "Training"  # Training, Predict, CV, Hyperparameter, Ensemble
  model: "CGCNN_demo"   # Model architecture to use
  job_name: "test_run"
  parallel: "True"      # Use multi-GPU if available
  save_model: "True"
  
Processing:
  data_path: "data/"
  data_format: "json"
  graph_max_radius: 8.0
  
Training:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  target_index: 0      # Column in targets.csv to predict
  loss: "mse"
  
Models:
  CGCNN_demo:
    epochs: 300
    batch_size: 64
    lr: 0.001
    gc_count: 4
    dropout_rate: 0.2
```

### 3. Training

Basic training:
```bash
python main.py --config_path config.yml --run_mode Training --model CGCNN_demo
```

Hyperparameter optimization:
```bash
python main.py --config_path config.yml --run_mode Hyperparameter --model CGCNN_demo
```

Cross-validation:
```bash
python main.py --config_path config.yml --run_mode CV --model CGCNN_demo
```

## Available Models

| Model | Description |
|-------|-------------|
| CGCNN | Crystal Graph Convolutional Neural Networks |
| GAT-GNN | Graph Attention Networks |
| MEGNet | Materials Energy Graph Network |
| SchNet | Quantum chemistry modeling |
| MPNN | Message Passing Neural Networks |
| GCN | Graph Convolutional Networks |
| VAE | Variational Autoencoders |
| Cycle-GAN | Domain transfer |
| Quantum-GAN | Quantum property modeling |
| Descriptor NN | Feature extraction |

## Advanced Usage

### Ensemble Learning

1. Configure ensemble in config.yml:
```yaml
Job:
  ensemble_list: "CGCNN_demo,MEGNet_demo,SchNet_demo"
```

2. Run ensemble:
```bash
python main.py --config_path config.yml --run_mode Ensemble
```

### Feature Analysis
```bash
python main.py --config_path config.yml --run_mode Analysis --model_path "models/best_model.pth"
```

### Custom Training
```python
from ME_Deep_Learning import models, training

model = models.get_model("CGCNN_demo")
trainer = training.ModelTrainer(model, config)
trainer.train()
```

## Output Structure
```
results/
├── models/             # Saved model checkpoints
├── predictions/        # Model predictions
├── analysis/          # Feature analysis plots
└── logs/              # Training logs
```

## Citation

```bibtex
@article{cette2024okcomputer,
    title={OK_Computer: A Deep Learning Framework for Materials Discovery},
    author={Cette AI}
   
}
```



## Contact
cette@ai.com

---
Made with ♥️ by Cette AI
