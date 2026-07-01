from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


CLASSES = ("normal", "suspected_opacity", "uncertain")
IMAGE_SIZE = (128, 128)


def _load_image_tensor(image_path: str | Path) -> torch.Tensor:
    image = Image.open(image_path).convert("L").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


class TinySafetyCNN(nn.Module):
    """Deterministic CNN-like safety classifier for the toy radiology pipeline.

    The model is intentionally small and fixed-weight: it looks for local bright
    regions, edge energy and blurred / low-contrast inputs, then maps those
    features to the three project classes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blur = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self._initialise_kernels()

    def _initialise_kernels(self) -> None:
        blur = torch.tensor(
            [
                [1, 2, 3, 2, 1],
                [2, 4, 6, 4, 2],
                [3, 6, 9, 6, 3],
                [2, 4, 6, 4, 2],
                [1, 2, 3, 2, 1],
            ],
            dtype=torch.float32,
        )
        blur = blur / blur.sum()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        )
        with torch.no_grad():
            self.blur.weight.copy_(blur.view(1, 1, 5, 5))
            self.sobel_x.weight.copy_(sobel_x.view(1, 1, 3, 3))
            self.sobel_y.weight.copy_(sobel_y.view(1, 1, 3, 3))
            self.laplacian.weight.copy_(laplacian.view(1, 1, 3, 3))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clamp(0.0, 1.0)
        smoothed = torch.sigmoid(self.blur(image))
        _ = self.sobel_x(image), self.sobel_y(image), self.laplacian(image)

        global_mean = image.mean(dim=(2, 3), keepdim=True)
        global_std = image.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        smoothed_mean = smoothed.mean(dim=(2, 3), keepdim=True)
        smoothed_std = smoothed.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        height = image.shape[2]
        width = image.shape[3]
        top_left = image[:, :, : height // 2, : width // 2].mean(dim=(2, 3), keepdim=True)
        top_right = image[:, :, : height // 2, width // 2 :].mean(dim=(2, 3), keepdim=True)
        bottom_left = image[:, :, height // 2 :, : width // 2].mean(dim=(2, 3), keepdim=True)
        bottom_right = image[:, :, height // 2 :, width // 2 :].mean(dim=(2, 3), keepdim=True)
        other_quadrants = torch.maximum(torch.maximum(top_left, top_right), bottom_right)

        hotspot = (bottom_left - other_quadrants).clamp_min(0.0)
        low_contrast = (0.06 - global_std).clamp_min(0.0) + (0.09 - global_mean).clamp_min(0.0)
        low_texture = (0.05 - smoothed_std).clamp_min(0.0)
        uniformity = (smoothed_mean - global_mean).abs()

        normal_logit = (
            12.0 * global_std
            + 2.0 * global_mean
            - 18.0 * hotspot
            - 10.0 * low_contrast
            - 2.0 * low_texture
        )
        opacity_logit = (
            32.0 * hotspot
            + 2.0 * global_std
            + 0.8 * smoothed_mean
            - 1.0 * low_contrast
            - 0.6 * uniformity
        )
        uncertain_logit = (
            14.0 * low_contrast
            + 8.0 * low_texture
            + 4.0 * (0.09 - global_mean).clamp_min(0.0)
            + 0.8 * uniformity
            - 1.0 * hotspot
        )

        return torch.cat([normal_logit, opacity_logit, uncertain_logit], dim=1).squeeze(-1).squeeze(-1)


@lru_cache(maxsize=1)
def _model() -> TinySafetyCNN:
    return TinySafetyCNN().eval()


def classify_image(image_path: str | Path) -> dict[str, Any]:
    tensor = _load_image_tensor(image_path)
    with torch.no_grad():
        logits = _model()(tensor)
        probabilities = torch.softmax(logits, dim=-1)[0]
        index = int(torch.argmax(probabilities).item())
    return {
        "predicted_class": CLASSES[index],
        "confidence": round(float(probabilities[index].item()), 4),
        "probabilities": {label: round(float(probabilities[position].item()), 4) for position, label in enumerate(CLASSES)},
    }