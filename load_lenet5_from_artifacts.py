"""Utility script to rebuild the best LeNet-5 model found via GridSearchCV."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn


DEFAULT_IMG_SIZE = 32
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
PARAMS_JSON = ARTIFACTS_DIR / "lenet5_grid_best_params.json"
WEIGHTS_PATH = ARTIFACTS_DIR / "lenet5_grid_best_weights.pt"


def load_metadata(path: Path = PARAMS_JSON) -> Dict[str, Any]:
    if not path.exists():
        msg = f"Missing metadata JSON at {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class LeNet5(nn.Module):
    """Must mirror the definition used during training so weights line up."""

    def __init__(self, input_channels: int = 1, num_classes: int = 10, img_size: int = DEFAULT_IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, img_size, img_size)
            features = self._forward_features(dummy)
            flatten_dim = features.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def rebuild_model(weights_path: Path = WEIGHTS_PATH) -> nn.Module:
    metadata = load_metadata()
    model_kwargs = metadata.get("model_kwargs", {})
    model = LeNet5(**model_kwargs)
    if not weights_path.exists():
        msg = f"Missing weight file at {weights_path}"
        raise FileNotFoundError(msg)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model with params:")
    print(json.dumps(metadata["best_params"], indent=2))
    print(f"CV score: {metadata['cv_score']:.4f}")
    return model


if __name__ == "__main__":
    _ = rebuild_model()
    print("Model is ready for inference.")
