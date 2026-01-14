# AI Coding Instructions for neumre-projekt

## Project Overview

This is a **comparative neural network study** implementing and benchmarking two model architectures (Feed-Forward NN and CNN) on the Fashion MNIST dataset. The project systematically evaluates hyperparameter tuning effects across both models.

**Key Goal**: Document performance differences between fully-connected and convolutional approaches through baseline training, systematic hyperparameter variations (3-5 per model), and detailed performance analysis.

## Architecture & Structure

### Main Components

- **[projekt.ipynb](projekt.ipynb)**: Primary deliverable - Jupyter notebook containing:
  - Data loading (FashionMNIST via torchvision)
  - Batch processing setup (batch_size=64)
  - Data visualization with labels mapping
  - Model implementations and training loops (to be expanded)

- **[PROJECT DOCS.md](PROJECT DOCS.md)**: Complete specifications including:
  - Required model comparison metrics (Accuracy, Precision, Recall, F1)
  - Hyperparameter tuning strategy (Regularization, Learning Rate, optional: batch size, epochs, optimizer)
  - Mandatory visualizations (GRAD-CAM for CNN, ROC curves, confusion matrices)
  - Per-class metrics required

### Data Pipeline

```
FashionMNIST (10 classes) → ToTensor() transform → DataLoader(batch_size=64)
├── Training: 60,000 images
└── Test: 10,000 images
```

Fashion MNIST labels (0-9): T-Shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot

## Key Patterns & Conventions

### Model Implementation Pattern

- **Baseline first**: Always train the default model to establish metrics before hyperparameter variations
- **Model variations**: Create 3-5 distinct versions per model with documented parameter differences
- **Parallel comparison**: Each model tuned independently; results aggregated for final analysis

### Hyperparameter Tuning Strategy

For each model variation, modify **one or more** of these parameters:
- Regularization strength (L1/L2)
- Learning rate
- (Optional) Batch size, epochs, optimizer type

Track results in a comparison structure for sensitivity analysis.

### Required Metrics Structure

Every trained model must report:
- Accuracy, Precision, Recall, F1-Score
- Per-class metrics (confusion matrix)
- Training/validation loss curves
- For CNN: GRAD-CAM visualizations, ROC curves/AUC

## Critical Dependencies & Imports

**Framework**: PyTorch 2.9.0 + torchvision 0.24.0

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
```

**Optional for visualizations**:
- scikit-learn (ROC curves, confusion matrix utilities)
- GRAD-CAM libraries (if doing explainability)

## Development Workflow

1. **Local Development**: Python + PyTorch with CUDA support (nvidia packages pre-configured)
2. **Colab Fallback**: Project explicitly supports Google Colab deployment for GPU acceleration
3. **Environment**: Full Jupyter stack configured (JupyterLab 4.4.9, IPython 8.37.0)

### Running the Notebook

Execute cells sequentially. Cell structure:
- Cell 1: Imports
- Cell 2: Data download & loading
- Cell 3: DataLoader verification
- Cell 4: Visualization examples
- Cell 5+: Add model implementations here

## Important Patterns to Follow

- **Data storage**: Always use `root="data"` path for FashionMNIST to match existing structure
- **Batch verification**: Check dataloader output shapes before model development
- **Class mapping**: Use provided `labels_map` dictionary for visualization consistency
- **GPU support**: Code should work with both CPU and CUDA (use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`)

## When Adding Model Code

- Implement Feed-Forward NN from scratch (no pre-trained models)
- CNN can use established architecture (LeNet variant or custom)
- Both models must accept (batch_size, 1, 28, 28) tensor input
- Implement separate train/test loops for reproducibility
- Track metrics per epoch for visualization

---

**Status**: Project in active development. Notebook structure established; model implementations and comparative analysis sections to follow.
