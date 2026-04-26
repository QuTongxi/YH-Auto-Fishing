from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from . import audit, config


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    index: int


class CompositeBuilder:
    def __init__(self) -> None:
        self.roi = config.ROI[config.SCREEN_MODE]
        self.size = config.MODEL_INPUT_SIZE

    @staticmethod
    def _resize_keep_aspect(img: Image.Image, target_width: int) -> Image.Image:
        w, h = img.size
        target_height = round(target_width * h / w)
        return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    def build(self, frame_bgr: np.ndarray) -> Image.Image:
        frame_rgb = frame_bgr[:, :, ::-1]
        src = Image.fromarray(frame_rgb).convert("RGB")

        width, height = src.size

        rects = [
            self.roi["FIGHT_BAR"],
            self.roi["WAIT_BANNER"],
            self.roi["SKILL_GROUP"],
        ]

        crops = []
        for rect in rects:
            x1, y1, x2, y2 = rect.to_pixels(width, height)
            crops.append(src.crop((x1, y1, x2, y2)))

        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))

        inner_width = round(self.size * 0.95)
        gap = round(self.size * 0.056)

        parts = [self._resize_keep_aspect(crop, inner_width) for crop in crops]
        total_h = sum(part.height for part in parts) + gap * (len(parts) - 1)

        y = max(0, (self.size - total_h) // 2)
        for part in parts:
            x = (self.size - part.width) // 2
            canvas.paste(part, (x, y))
            y += part.height + gap

        return canvas


class MobileNetClassifier:
    def __init__(self, model_path: Path = config.MOBILE_NET_MODEL_PATH) -> None:
        if config.TORCH_NUM_THREADS > 0:
            torch.set_num_threads(config.TORCH_NUM_THREADS)

        self.model_path = model_path
        self.device = self._select_device()
        self.builder = CompositeBuilder()

        checkpoint = torch.load(model_path, map_location=self.device)

        self.class_names = checkpoint["class_names"]
        num_classes = len(self.class_names)

        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        audit.normal(
            f"Loaded MobileNet model: {model_path} device={self.device} classes={self.class_names}"
        )

    @staticmethod
    def _select_device() -> torch.device:
        if config.MODEL_DEVICE == "cpu":
            return torch.device("cpu")
        if config.MODEL_DEVICE == "cuda":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def predict_image(self, image: Image.Image) -> Prediction:
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

        confidence, index = torch.max(probs, dim=0)
        idx = int(index.item())

        pred = Prediction(
            label=self.class_names[idx],
            confidence=float(confidence.item()),
            index=idx,
        )

        audit.prediction(pred.label, pred.confidence)
        return pred

    def predict_frame(self, frame_bgr: np.ndarray) -> Prediction:
        image = self.builder.build(frame_bgr)
        return self.predict_image(image)


class AsyncMobileNetClassifier:
    def __init__(self, classifier: MobileNetClassifier) -> None:
        self.classifier = classifier
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: Optional[Future[Prediction]] = None
        self.latest_prediction: Optional[Prediction] = None

    def submit(self, frame_bgr: np.ndarray) -> None:
        if self.future is not None and not self.future.done():
            return

        if self.future is not None and self.future.done():
            self.latest_prediction = self.future.result()
            self.future = None

        frame_copy = frame_bgr.copy()
        self.future = self.executor.submit(self.classifier.predict_frame, frame_copy)

    def latest(self) -> Optional[Prediction]:
        if self.future is not None and self.future.done():
            self.latest_prediction = self.future.result()
            self.future = None
        return self.latest_prediction

    def predict_now(self, frame_bgr: np.ndarray) -> Prediction:
        return self.classifier.predict_frame(frame_bgr)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)