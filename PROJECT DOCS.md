# Fashion MNIST Classification: Comparative Model Analysis

## Project Overview

This project implements and compares two different neural network approaches for image classification on the Fashion MNIST dataset. The goal is to analyze the performance, accuracy, and behavior of each model architecture through systematic experimentation and hyperparameter tuning.

---

## Dataset

### Fashion MNIST

- **Description**: A dataset of grayscale images (28x28 pixels) containing 10 classes of fashion items
- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Purpose**: Used for training and evaluating both model architectures

---

## Model Architectures

We will implement and compare **two distinct models**:

### 1. Feed-Forward Neural Network

- **Type**: Custom implementation
- **Architecture**: Fully connected layers
- **Implementation**: Built from scratch to understand fundamental concepts

### 2. Convolutional Neural Network (CNN)

- **Type**: Pre-implemented architecture
- **Architecture**: Convolutional layers with pooling
- **Implementation**: Selected from established CNN architectures (e.g., LeNet, custom CNN)

---

## Methodology & Experimentation

### Training Workflow

For **each model independently**, we will follow this systematic approach:

1. **Baseline Training**

   - Train the model using default parameters
   - Establish baseline performance metrics
   - Document initial accuracy and loss

2. **Hyperparameter Tuning**

   - Create **3-5 variations** of the model with different parameter configurations
   - Systematically modify hyperparameters to observe impact

3. **Parameter Variations**

   - **Regularization factor** (L1/L2 regularization strength)
   - **Learning rate** (step size for gradient descent)
   - Additional parameters as needed (batch size, epochs, optimizer type)

4. **Comparison & Analysis**
   - Compare all variations against the baseline model
   - Identify optimal configurations
   - Document performance improvements or degradation

---

## Documentation Requirements

### Model Comparison

Document the **main differences between the two approaches**:

- **Architecture & Mechanism**

  - How each model processes input data
  - Key structural differences (fully connected vs. convolutional layers)
  - Computational complexity

- **Performance Metrics**

  - Accuracy
  - Precision
  - Recall
  - F1 Score

- **Challenges & Insights**

  - Obstacles encountered during implementation
  - Structural issues specific to each architecture
  - Overfitting/underfitting observations

- **Hyperparameter Impact**
  - Detailed comparison of results after tuning
  - Sensitivity analysis for each parameter
  - Optimal configurations identified

---

## Final Results & Visualization

### Evaluation Metrics

1. **GRAD-CAM (Gradient-weighted Class Activation Mapping)**

   - Visual explanation of which regions in the image the model focuses on
   - Interpretability analysis for CNN predictions
   - Understanding model decision-making process

2. **ROC Curve & AUC (Receiver Operating Characteristic / Area Under Curve)**

   - Classification performance across all thresholds
   - Per-class ROC curves for multiclass classification
   - Overall model discriminative ability

3. **Additional Metrics**
   - Confusion matrix
   - Precision, Recall, F1-Score per class
   - Training/validation loss curves

---

## Technical Specifications

### Development Environment

- **Primary**: Local development with Python, PyTorch/TensorFlow
- **Backup**: Google Colab for computationally intensive training
  - Use when local setup is too slow or lacks GPU support
  - Leverage free GPU/TPU resources for faster experimentation

### Version Control & Collaboration

- **Platform**: GitHub
- **Purpose**: Code versioning, collaboration, and documentation
- **Branch Strategy**: Feature branches for different experiments

## Optional Enhancements

- **CI/CD Pipeline**: Setup auto-deploy to Google Colab for seamless cloud training

---

## Implementation Snapshot (Jan 13, 2026)

- End-to-end experimentation lives in [run_experiments.py](run_experiments.py). It orchestrates data loading, model construction, training loops, evaluation, and artifact exports.
- All raw metrics and visualizations are saved to the `artifacts/` tree:
   - CSV summary for quick comparisons: [artifacts/results_summary.csv](artifacts/results_summary.csv)
   - Per-variant JSON metrics (macro + per-class): [artifacts/metrics](artifacts/metrics)
   - Plots (confusion matrices, ROC curves, training curves, Grad-CAM heatmap): [artifacts/plots](artifacts/plots)
- Data pipeline: FashionMNIST train set split into 55k train / 5k validation via `torch.utils.data.random_split`, batch size 64, `ToTensor()` transform only.
- Training uses `AdamW` with seeded reproducibility (seed 42) and best-checkpoint selection via validation accuracy.

## Experiment Matrix — Feed-Forward Network Variants

| Variant | Hidden Dims | Dropout | LR | Weight Decay | Epochs | Test Acc | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ffnn_baseline | [256, 128] | 0.30 | 1.0e-3 | 1.0e-4 | 6 | **87.46%** | 0.874 |
| ffnn_lr_decay | [512, 256, 128] | 0.40 | 1.5e-3 | 5.0e-4 | 8 | 86.85% | 0.868 |
| ffnn_compact | [256, 64] | 0.20 | 8.0e-4 | 1.0e-5 | 6 | 87.05% | 0.870 |
| ffnn_dropout_sweep | [512, 256, 64] | 0.50 | 1.0e-3 | 3.0e-4 | 8 | 86.42% | 0.862 |

**Notes**

- Accuracy plateaued near 87% despite deeper stacks; heavier dropout hurt `Shirt` recall (0.60) more than it reduced overfitting.
- `ffnn_baseline` retained the cleanest loss curves ([artifacts/plots/ffnn_baseline_curves.png](artifacts/plots/ffnn_baseline_curves.png)) and was selected as the reference FFNN model.

## Experiment Matrix — CNN Variants

| Variant | Channels | Dropout | LR | Weight Decay | Epochs | Test Acc | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_baseline | (32, 64, 128) | 0.30 | 1.0e-3 | 1.0e-4 | 6 | 91.97% | 0.920 |
| cnn_deep | (48, 96, 192) | 0.35 | 8.0e-4 | 5.0e-4 | 8 | 91.69% | 0.917 |
| cnn_light | (24, 48, 96) | 0.25 | 1.2e-3 | 1.0e-5 | 6 | 91.67% | 0.917 |
| cnn_dropout_sweep | (32, 64, 128) | 0.45 | 7.0e-4 | 8.0e-4 | 9 | **92.57%** | **0.925** |

**Notes**

- CNNs surpassed FFNNs by ~5 pp accuracy thanks to spatial feature extraction; the heavier dropout variant generalized best without sacrificing convergence ([artifacts/plots/cnn_dropout_sweep_curves.png](artifacts/plots/cnn_dropout_sweep_curves.png)).
- `cnn_dropout_sweep` yields macro precision/recall of 0.926, with the hardest class (`Shirt`) still lagging (precision 0.79, recall 0.769 as seen in [artifacts/metrics/cnn_dropout_sweep.json](artifacts/metrics/cnn_dropout_sweep.json)).

## Key Comparative Insights

- **Architecture effect**: Switching from flattened inputs to convolutions boosts macro F1 from 0.874 → 0.925, primarily by improving `Pullover`, `Coat`, and `Shirt` recognition.
- **Regularization**: Moderate dropout (0.30–0.35) stabilizes FFNNs, while CNNs benefited from a stronger 0.45 rate combined with weight decay 8e-4. Excess dropout in FFNNs reduced capacity before benefits materialized.
- **Class-level behavior**: Both families already exceed 0.96 F1 for `Trouser`, `Sandal`, `Bag`, and `Ankle Boot`. The main confusion pockets are `Pullover`↔`Coat` and `Shirt`↔`T-Shirt`, visible in [artifacts/plots/ffnn_baseline_confusion.png](artifacts/plots/ffnn_baseline_confusion.png) and [artifacts/plots/cnn_dropout_sweep_confusion.png](artifacts/plots/cnn_dropout_sweep_confusion.png).
- **Interpretability**: Grad-CAM heatmaps ([artifacts/plots/cnn_dropout_sweep_gradcam.png](artifacts/plots/cnn_dropout_sweep_gradcam.png)) show the CNN focusing on collars and footwear edges, matching expected discriminative regions.
- **ROC/AUC**: Per-class ROC curves for the best FFNN and CNN live in [artifacts/plots/ffnn_baseline_roc.png](artifacts/plots/ffnn_baseline_roc.png) and [artifacts/plots/cnn_dropout_sweep_roc.png](artifacts/plots/cnn_dropout_sweep_roc.png); CNN curves hug the top-left corner with AUC ≥ 0.98 for most classes.

## Visualization Gallery

| Artifact | Description |
| --- | --- |
| [artifacts/plots/ffnn_baseline_curves.png](artifacts/plots/ffnn_baseline_curves.png) | Training vs. validation loss/accuracy for the best FFNN.
| [artifacts/plots/ffnn_baseline_confusion.png](artifacts/plots/ffnn_baseline_confusion.png) | FFNN confusion matrix (test split).
| [artifacts/plots/ffnn_baseline_roc.png](artifacts/plots/ffnn_baseline_roc.png) | Per-class ROC curves for FFNN baseline.
| [artifacts/plots/cnn_dropout_sweep_curves.png](artifacts/plots/cnn_dropout_sweep_curves.png) | CNN training curves (best variant).
| [artifacts/plots/cnn_dropout_sweep_confusion.png](artifacts/plots/cnn_dropout_sweep_confusion.png) | CNN confusion matrix.
| [artifacts/plots/cnn_dropout_sweep_roc.png](artifacts/plots/cnn_dropout_sweep_roc.png) | CNN ROC curves.
| [artifacts/plots/cnn_dropout_sweep_gradcam.png](artifacts/plots/cnn_dropout_sweep_gradcam.png) | Grad-CAM overlay explaining CNN focus areas.

## Reproduction Checklist

1. Verify dependencies (PyTorch 2.9, torchvision 0.24, matplotlib 3.9, scikit-learn 1.5) are installed in the repo virtualenv.
2. Run `python run_experiments.py` from the repository root (`neumre-projekt/`). Training all eight variants on CPU takes ~10 minutes.
3. Inspect aggregate metrics in [artifacts/results_summary.csv](artifacts/results_summary.csv) and drill into per-class stats via JSON files in [artifacts/metrics](artifacts/metrics).
4. Embed or export relevant plots from [artifacts/plots](artifacts/plots) when preparing presentations or reports.

## Open Questions / Next Steps

- Could lightweight augmentations (random horizontal flips, slight rotations) raise `Shirt` recall without hurting structured classes such as `Trouser`?
- The current pipeline relies on manual hyperparameter grids. Consider integrating Optuna or Ray Tune for smarter search once GPU time is available.
- Extend interpretability beyond Grad-CAM by logging misclassified samples per class to understand failure modes more deeply.
