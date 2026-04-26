"""
python src/train.py --data ./dataset --out ./runs/fishing_mobilenetv3 --epochs 30 --batch-size 32 --size 320
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]

        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(
    samples: list[tuple[str, int]],
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    rng = random.Random(seed)

    by_class: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for sample in samples:
        _, label = sample
        by_class[label].append(sample)

    train_samples = []
    test_samples = []

    for label, items in by_class.items():
        rng.shuffle(items)

        test_count = max(1, round(len(items) * test_ratio))

        test_samples.extend(items[:test_count])
        train_samples.extend(items[test_count:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    return train_samples, test_samples


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
    else:
        weights = None

    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = True


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits.detach(), labels) * batch_size
        total_count += batch_size

    return total_loss / total_count, total_acc / total_count


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for images, labels in tqdm(loader, desc="test", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits, labels) * batch_size
        total_count += batch_size

    return total_loss / total_count, total_acc / total_count


def make_class_weights(samples: list[tuple[str, int]], num_classes: int) -> torch.Tensor:
    counts = Counter(label for _, label in samples)
    total = sum(counts.values())

    weights = []
    for label in range(num_classes):
        count = counts[label]
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./dataset", help="Dataset root directory.")
    parser.add_argument("--out", default="./runs/fishing_mobilenetv3", help="Output directory.")
    parser.add_argument("--size", type=int, default=320, help="Input image size.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = datasets.ImageFolder(root=data_dir)

    class_names = base_dataset.classes
    class_to_idx = base_dataset.class_to_idx
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {num_classes}")

    train_samples, test_samples = stratified_split(
        samples=base_dataset.samples,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("Classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    print()
    print(f"Total images: {len(base_dataset.samples)}")
    print(f"Train images: {len(train_samples)}")
    print(f"Test images:  {len(test_samples)}")
    print()

    train_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = ImagePathDataset(train_samples, transform=train_transform)
    test_dataset = ImagePathDataset(test_samples, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    model = build_model(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
    )
    model.to(device)

    if args.freeze_epochs > 0:
        freeze_backbone(model)

    class_weights = make_class_weights(train_samples, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_acc = 0.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    labels_path = out_dir / "labels.json"
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": class_names,
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(i): name for i, name in enumerate(class_names)},
                "input_size": args.size,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.finetune_lr,
                weight_decay=1e-4,
            )
            print("Backbone unfrozen for fine-tuning.")
            print()

        train_loss, train_acc = run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = run_eval_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

        checkpoint = {
            "model_name": "mobilenet_v3_small",
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "input_size": args.size,
            "epoch": epoch,
            "test_acc": test_acc,
        }

        torch.save(checkpoint, last_path)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path} acc={best_acc:.4f}")

    print()
    print("Training finished.")
    print(f"Best test acc: {best_acc:.4f}")
    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    main()