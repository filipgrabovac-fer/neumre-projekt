# Fashion MNIST Classification: Comparative CNN Analysis

## Project Documentation

### 1. Project Overview

This project implements and compares two distinct Convolutional Neural Network (CNN) architectures for image classification on the Fashion MNIST dataset. The primary objective is to evaluate how a Simplified Custom CNN performs against the industry-standard LeNet-5 architecture, focusing on the trade-offs between architectural complexity, computational efficiency, and classification accuracy.

---

### 2. Dataset Specifications

- **Name**: Fashion MNIST
- **Description**: A dataset of grayscale images containing 10 classes of fashion items
- **Input Dimensions**: 28x28 pixels (Grayscale)
- **Size**: 60,000 Training images / 10,000 Test images
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

### 3. Model Architectures

We will implement and compare two distinct models using **PyTorch Standard** (`torch.nn.Module`).

#### Model A: Simplified Custom CNN

- **Type**: Custom Implementation
- **Architecture**: A lightweight CNN designed "from scratch"
- **Design Constraints**: Must be simpler than LeNet-5 (e.g., fewer layers or filters)
- **Purpose**: To test the efficiency of a minimal architecture and understand the impact of reducing model capacity

#### Model B: LeNet-5 (Reference Model)

- **Type**: Established Architecture
- **Architecture**: The classic 1998 LeCun architecture (2 Convolutional layers, Average/Max Pooling, 3 Fully Connected layers)
- **Purpose**: To serve as the robust performance benchmark

---

### 4. Methodology & Experimentation

Both models will undergo identical training and evaluation pipelines to ensure a fair comparison.

#### 4.1. Technology Stack

- **Language**: Python 3.8+
- **Framework**: PyTorch
- **Hyperparameter Tuning**: Scikit-Learn (GridSearchCV) via a wrapper (e.g., skorch or custom wrapper)
- **Environment**: Dependencies fixed via `requirements.txt`

#### 4.2. Training Workflow

1. **Baseline Training**: Train both models with default parameters to establish initial metrics

2. **Hyperparameter Tuning**: Use GridSearchCV to optimize:
   - Learning Rate
   - Batch Size
   - Regularization Strength (L2 / Weight Decay)

   > **Note**: Both models must use identical regularization techniques.

---

### 5. Evaluation & Visualization Requirements

#### 5.1. Performance Metrics

- Accuracy (Global)
- Confusion Matrix
- Precision, Recall, F1-Score (Per-class)

#### 5.2. ROC Analysis (One-vs-Rest)

- **Requirement**: Plot 10 separate ROC curves for each model (one for each class vs. the rest)
- **Goal**: To visualize the specific classes where the Simplified Model might be losing discriminative power compared to LeNet-5

#### 5.3. Interpretability (GRAD-CAM)

- **Technique**: Gradient-weighted Class Activation Mapping
- **Application**: Apply to **BOTH** models (Custom and LeNet-5)
- **Goal**: Visually compare the "heatmaps" of what the models look at. Does the simplified model focus on the same features as the LeNet-5 model?

---

## Independent Task Groups

The project is divided into **5 independent modules**. Each group works on a specific file or set of functions, which will be integrated via the main Jupyter Notebook.

### Group 1: Data Pipeline & Environment Setup

**Responsibility**: Ensure data is ready and the environment is stable for all other groups.

- **Task 1.1**: Create `requirements.txt` locking versions for `torch`, `torchvision`, `scikit-learn`, `numpy`, `matplotlib`
- **Task 1.2**: Implement data loading functions (Download dataset, apply transformations: Normalization, Resizing if needed)
- **Task 1.3**: Implement `get_data_loaders(batch_size)` function that returns Train/Val/Test loaders

**Deliverable**: `data_setup.py`

---

### Group 2: Model A (Custom Simplified CNN) Implementation

**Responsibility**: Build the experimental lightweight model.

- **Task 2.1**: Implement the `SimpleCNN` class using `torch.nn.Module`
- **Task 2.2**: Ensure the architecture is strictly simpler than LeNet-5 (e.g., 1 Conv block, fewer filters)
- **Task 2.3**: Implement a basic `train_step` function for this model to verify it learns

**Deliverable**: `model_custom.py`

---

### Group 3: Model B (LeNet-5) & Tuning Framework

**Responsibility**: Build the reference model and the tuning mechanism.

- **Task 3.1**: Implement the `LeNet5` class using `torch.nn.Module` (exact specs: 2 Conv layers, 3 FC layers)
- **Task 3.2**: Implement the wrapper logic to make PyTorch models compatible with `sklearn.model_selection.GridSearchCV`
- **Task 3.3**: Define the hyperparameter grid (`params` dictionary)

**Deliverable**: `model_lenet.py` and `tuning_utils.py`

---

### Group 4: Advanced Metrics (ROC & AUC)

**Responsibility**: Handle the complex multi-class evaluation math.

- **Task 4.1**: Implement logic to generate probabilities (Softmax) from model outputs
- **Task 4.2**: Implement Binarization of labels for One-vs-Rest (OvR) calculation
- **Task 4.3**: Create a plotting function that generates a figure with 10 separate ROC curves (one per class)

**Deliverable**: `evaluation_metrics.py`

---

### Group 5: Interpretability (GRAD-CAM)

**Responsibility**: Visualizing model behavior.

- **Task 5.1**: Implement the GRAD-CAM hook logic (registering hooks on the final convolutional layer to capture gradients)
- **Task 5.2**: Create a visualization function that overlays the heatmap onto the original grayscale image
- **Task 5.3**: Create a comparison view: **Input Image | Simple CNN Heatmap | LeNet-5 Heatmap**

**Deliverable**: `visualization.py`
