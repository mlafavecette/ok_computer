"""
Materials Graph Neural Network Models

A comprehensive collection of graph neural network architectures optimized for
materials science applications.

Each model implements different approaches to structure-property prediction:
- GCN: Basic graph convolutions
- MPNN: Message passing architectures
- SchNet: Continuous filter convolutions
- CGCNN: Crystal-specific convolutions
- MEGNet: Multi-level message passing
- GATGNN: Graph attention mechanisms
- Descriptor-based: SOAP and SM classical approaches
"""

from .gcn import GCN
from .mpnn import MPNN
from .schnet import SchNet
from .cgcnn import CGCNN
from .megnet import MEGNet
from .gatgnn import GATGNN
from .deep_gatgnn import DeepGATGNN
from .super_cgcnn import SuperCGCNN
from .super_megnet import SuperMEGNet
from .super_schnet import SuperSchNet
from .super_mpnn import SuperMPNN
from .descriptor_nn import SOAP, SM

__all__ = [
    "GCN",
    "MPNN",
    "SchNet",
    "CGCNN",
    "MEGNet",
    "SOAP",
    "SM",
    "GATGNN",
    "DeepGATGNN",
    "SuperCGCNN",
    "SuperMEGNet",
    "SuperSchNet",
    "SuperMPNN"
]