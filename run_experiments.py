import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

LABELS = [
    "T-Shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
]

ARTIFACT_ROOT = Path("artifacts")
PLOTS_DIR = ARTIFACT_ROOT / "plots"
METRICS_DIR = ARTIFACT_ROOT / "metrics"
ARTIFACT_ROOT.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeedForwardNet(nn.Module):
    def __init__(self, hidden_dims=None, dropout: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        layers = []
        input_dim = 28 * 28
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = dim
        layers.append(nn.Linear(input_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, channels=None, dropout: float = 0.3):
        super().__init__()
        channels = channels or (32, 64, 128)
        c1, c2, c3 = channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.target_conv = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.block3 = nn.Sequential(
            self.target_conv,
            nn.BatchNorm2d(c3),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c3 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_dataloaders(batch_size: int = 64, val_split: int = 5000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

    train_size = len(train_dataset) - val_split
    val_size = val_split
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, criterion, device, collect: bool = False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_targets, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            if collect:
                all_targets.append(labels.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(torch.softmax(logits, dim=1).cpu())
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    if collect:
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()
    else:
        y_true = None
        y_pred = None
        y_prob = None
    return avg_loss, accuracy, y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
    per_class = {}
    for idx, label in enumerate(LABELS):
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return metrics, per_class, cm.tolist()


def plot_training_curves(history, title, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"Loss • {title}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title(f"Accuracy • {title}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_conf_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_roc_curves(y_true, y_prob, classes, title, save_path):
    y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, class_name in enumerate(classes):
        if y_true_bin[:, idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def generate_grad_cam(model, data_loader, classes, title, save_path, device):
    model.eval()

    chosen_image = None
    chosen_label = None
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
        match_mask = preds == labels
        if match_mask.any():
            idx = match_mask.nonzero(as_tuple=False)[0].item()
            chosen_image = images[idx].unsqueeze(0)
            chosen_label = labels[idx].item()
            break
    if chosen_image is None:
        return

    activations = []
    gradients = []

    def forward_hook(module, input_, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = model.target_conv.register_forward_hook(forward_hook)
    handle_b = model.target_conv.register_full_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(chosen_image)
    score = logits[0, chosen_label]
    score.backward()

    activation = activations[-1].detach()
    gradient = gradients[-1].detach()
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=chosen_image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    image_np = chosen_image.squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_np, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.4)
    ax.axis("off")
    ax.set_title(f"{title}: {classes[chosen_label]}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    handle_f.remove()
    handle_b.remove()


def train_variant(model, spec, loaders, device):
    lr = spec.get("lr", 1e-3)
    weight_decay = spec.get("weight_decay", 0.0)
    epochs = spec.get("epochs", 8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_state = {"epoch": 0, "accuracy": 0.0, "state_dict": copy.deepcopy(model.state_dict())}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        val_loss, val_acc, _, _, _ = evaluate(model, loaders["val"], criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_state["accuracy"]:
            best_state["accuracy"] = val_acc
            best_state["epoch"] = epoch
            best_state["state_dict"] = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state["state_dict"])
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(
        model, loaders["test"], criterion, device, collect=True
    )
    _, val_acc, _, _, _ = evaluate(model, loaders["val"], criterion, device, collect=False)

    return {
        "history": history,
        "best_epoch": best_state["epoch"],
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "model_state": copy.deepcopy(model.state_dict()),
    }


FEEDFORWARD_VARIANTS = [
    {
        "name": "ffnn_baseline",
        "architecture": "feedforward",
        "hidden_dims": [256, 128],
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 6,
    },
    {
        "name": "ffnn_lr_decay",
        "architecture": "feedforward",
        "hidden_dims": [512, 256, 128],
        "dropout": 0.4,
        "lr": 1.5e-3,
        "weight_decay": 5e-4,
        "epochs": 8,
    },
    {
        "name": "ffnn_compact",
        "architecture": "feedforward",
        "hidden_dims": [256, 64],
        "dropout": 0.2,
        "lr": 8e-4,
        "weight_decay": 1e-5,
        "epochs": 6,
    },
    {
        "name": "ffnn_dropout_sweep",
        "architecture": "feedforward",
        "hidden_dims": [512, 256, 64],
        "dropout": 0.5,
        "lr": 1e-3,
        "weight_decay": 3e-4,
        "epochs": 8,
    },
]

CNN_VARIANTS = [
    {
        "name": "cnn_baseline",
        "architecture": "cnn",
        "channels": (32, 64, 128),
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 6,
    },
    {
        "name": "cnn_deep",
        "architecture": "cnn",
        "channels": (48, 96, 192),
        "dropout": 0.35,
        "lr": 8e-4,
        "weight_decay": 5e-4,
        "epochs": 8,
    },
    {
        "name": "cnn_light",
        "architecture": "cnn",
        "channels": (24, 48, 96),
        "dropout": 0.25,
        "lr": 1.2e-3,
        "weight_decay": 1e-5,
        "epochs": 6,
    },
    {
        "name": "cnn_dropout_sweep",
        "architecture": "cnn",
        "channels": (32, 64, 128),
        "dropout": 0.45,
        "lr": 7e-4,
        "weight_decay": 8e-4,
        "epochs": 9,
    },
]


def build_model(spec):
    if spec["architecture"] == "feedforward":
        return FeedForwardNet(hidden_dims=spec.get("hidden_dims"), dropout=spec.get("dropout", 0.3))
    if spec["architecture"] == "cnn":
        return SimpleCNN(channels=spec.get("channels"), dropout=spec.get("dropout", 0.3))
    raise ValueError(f"Unknown architecture: {spec['architecture']}")


def save_metrics(name, architecture, config, result):
    metrics, per_class, cm = compute_metrics(result["y_true"], result["y_pred"], result["y_prob"])
    payload = {
        "name": name,
        "architecture": architecture,
        "config": config,
        "best_epoch": result["best_epoch"],
        "val_accuracy": result["val_accuracy"],
        "test_accuracy": result["test_accuracy"],
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
    with open(METRICS_DIR / f"{name}.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return metrics, per_class, cm


def export_summary(rows):
    summary_path = ARTIFACT_ROOT / "results_summary.csv"
    fieldnames = [
        "architecture",
        "name",
        "epochs",
        "lr",
        "weight_decay",
        "dropout",
        "params",
        "best_epoch",
        "val_accuracy",
        "test_accuracy",
        "precision",
        "recall",
        "f1",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(batch_size=64)

    all_specs = FEEDFORWARD_VARIANTS + CNN_VARIANTS
    results = []
    summary_rows = []

    for spec in all_specs:
        name = spec["name"]
        architecture = spec["architecture"]
        print(f"Running {name} ({architecture}) on {device}")
        model = build_model(spec).to(device)
        start = time.time()
        variant_result = train_variant(model, spec, loaders, device)
        duration = time.time() - start
        metrics, _, _ = save_metrics(name, architecture, spec, variant_result)
        params = count_params(model)
        summary_rows.append(
            {
                "architecture": architecture,
                "name": name,
                "epochs": spec.get("epochs", 0),
                "lr": spec.get("lr", 0.0),
                "weight_decay": spec.get("weight_decay", 0.0),
                "dropout": spec.get("dropout", 0.0),
                "params": params,
                "best_epoch": variant_result["best_epoch"],
                "val_accuracy": variant_result["val_accuracy"],
                "test_accuracy": variant_result["test_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
        print(
            f"{name}: val_acc={variant_result['val_accuracy']:.4f} test_acc={variant_result['test_accuracy']:.4f} duration={duration/60:.2f} min",
        )
        results.append({
            "name": name,
            "architecture": architecture,
            "config": spec,
            "result": variant_result,
        })

    summary_path = export_summary(summary_rows)
    print(f"Saved summary to {summary_path}")

    def best_for(arch):
        arch_results = [r for r in results if r["architecture"] == arch]
        return max(arch_results, key=lambda item: item["result"]["test_accuracy"])

    best_ff = best_for("feedforward")
    best_cnn = best_for("cnn")

    for best in (best_ff, best_cnn):
        name = best["name"]
        history = best["result"]["history"]
        plot_training_curves(history, name, PLOTS_DIR / f"{name}_curves.png")
        plot_conf_matrix(
            best["result"]["y_true"],
            best["result"]["y_pred"],
            LABELS,
            f"Confusion • {name}",
            PLOTS_DIR / f"{name}_confusion.png",
        )
        plot_roc_curves(
            best["result"]["y_true"],
            best["result"]["y_prob"],
            LABELS,
            f"ROC • {name}",
            PLOTS_DIR / f"{name}_roc.png",
        )

    cnn_model = build_model(best_cnn["config"]).to(device)
    cnn_model.load_state_dict(best_cnn["result"]["model_state"])
    generate_grad_cam(
        cnn_model,
        loaders["test"],
        LABELS,
        f"Grad-CAM • {best_cnn['name']}",
        PLOTS_DIR / f"{best_cnn['name']}_gradcam.png",
        device,
    )


if __name__ == "__main__":
    main()
