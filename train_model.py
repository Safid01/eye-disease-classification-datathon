import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2

sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)

class EyeDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self.paths = self.df["proc_path"].values
        if not self.is_test:
            self.targets = self.df["target"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

        if self.is_test:
            item_id = self.df.loc[idx, "id"]
            return img, item_id
        else:
            target = self.targets[idx]
            return img, torch.tensor(target, dtype=torch.long)

def get_train_transforms(img_size=384):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.92, 1.08), translate_percent=(-0.05, 0.05), rotate=(-30, 30), p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_valid_transforms(img_size=384):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def train_epoch(model, dataloader, criterion, optimizer, device, use_autocast=True):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_autocast=True):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * targets.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.append(probs.float().cpu().numpy())

    val_loss = total_loss / len(all_targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.vstack(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return val_loss, acc, macro_f1, all_preds, all_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="convnext_small.fb_in22k_ft_in1k_384")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(42 + args.fold)

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {device} ({torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'CPU'})")

    metadata_path = Path("n:/Ai Dev Fest/data_384/train_metadata.csv")
    df = pd.read_csv(metadata_path)

    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    val_df = df[df["fold"] == args.fold].reset_index(drop=True)

    print(f"Fold {args.fold}: Train samples={len(train_df)}, Val samples={len(val_df)}")

    # Calculate smoothed class weights
    class_counts = train_df["target"].value_counts().sort_index().values
    total_samples = len(train_df)
    n_classes = len(class_counts)
    # Smoothed inverse frequency with power 0.4
    weights = (total_samples / (n_classes * class_counts)) ** 0.4
    weights = weights / weights.mean() # normalize so mean weight is 1.0
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Class weights:", np.round(weights, 2))

    train_dataset = EyeDataset(train_df, transform=get_train_transforms(args.img_size))
    val_dataset = EyeDataset(val_df, transform=get_valid_transforms(args.img_size))

    # DataLoader with num_workers=0 on Windows for stability, non_blocking=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"Creating model: {args.model_name} with num_classes=10...")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.08)

    # Optimizer with differential learning rate: backbone slightly lower, head slightly higher
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = Path(f"n:/Ai Dev Fest/{args.save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    clean_model_name = args.model_name.replace("/", "_")
    best_ckpt_path = save_dir / f"{clean_model_name}_fold{args.fold}_best.pth"

    best_macro_f1 = 0.0
    best_acc = 0.0

    print(f"\n--- Starting Training {args.model_name} (Fold {args.fold}) for {args.epochs} epochs ---")
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_time:.1f}s] - "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro-F1: {val_macro_f1:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model based on composite metric (Val Acc + Macro F1)
        metric = (val_acc + val_macro_f1) / 2.0
        best_metric = (best_acc + best_macro_f1) / 2.0 if best_macro_f1 > 0 else 0.0

        if metric > best_metric:
            best_acc = val_acc
            best_macro_f1 = val_macro_f1
            torch.save({
                "epoch": epoch,
                "model_name": args.model_name,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "fold": args.fold,
                "img_size": args.img_size
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path.name} (Acc: {val_acc:.4f}, Macro-F1: {val_macro_f1:.4f})")

    total_time = time.time() - start_total_time
    print(f"\nTraining Complete in {total_time/60:.2f} minutes!")
    print(f"Best Validation Accuracy: {best_acc:.4f}, Best Macro-F1: {best_macro_f1:.4f}")

if __name__ == "__main__":
    main()
