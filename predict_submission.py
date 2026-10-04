import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.stdout.reconfigure(line_buffering=True)

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.paths = self.df["proc_path"].values
        self.ids = self.df["id"].values

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

        item_id = self.ids[idx]
        return img, item_id

def get_base_transform(img_size=384):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_hflip_transform(img_size=384):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_vflip_transform(img_size=384):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

@torch.no_grad()
def predict_model_tta(model, df_test, img_size, device, batch_size=32):
    model.eval()
    
    transforms_list = [
        ("orig", get_base_transform(img_size)),
        ("hflip", get_hflip_transform(img_size)),
        ("vflip", get_vflip_transform(img_size))
    ]
    
    all_tta_probs = []
    
    for tta_name, transform in transforms_list:
        test_dataset = TestDataset(df_test, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        tta_probs = []
        ids = []
        for images, batch_ids in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            tta_probs.append(probs.float().cpu().numpy())
            ids.extend(batch_ids.numpy())
            
        tta_probs = np.vstack(tta_probs)
        all_tta_probs.append(tta_probs)
        print(f"  TTA {tta_name} prediction completed.")
        
    avg_probs = np.mean(all_tta_probs, axis=0)
    return ids, avg_probs

def generate_submission(checkpoint_paths, output_csv="n:/Ai Dev Fest/submission.csv", batch_size=32):
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Using device for inference: {device}")

    test_meta_path = Path("n:/Ai Dev Fest/data_384/test_metadata.csv")
    df_test = pd.read_csv(test_meta_path).sort_values("id").reset_index(drop=True)
    print(f"Loaded {len(df_test)} test samples (IDs {df_test['id'].min()} to {df_test['id'].max()}).")

    model_probs_list = []
    ids = None

    for ckpt_path in checkpoint_paths:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint {ckpt_path} not found! Skipping.")
            continue
            
        print(f"\nLoading checkpoint: {ckpt_path.name}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_name = checkpoint.get("model_name", "convnext_small.fb_in22k_ft_in1k_384")
        img_size = checkpoint.get("img_size", 384)
        val_acc = checkpoint.get("val_acc", 0.0)
        val_f1 = checkpoint.get("val_macro_f1", 0.0)
        print(f"Model: {model_name} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        model = timm.create_model(model_name, pretrained=False, num_classes=10)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        ids, avg_probs = predict_model_tta(model, df_test, img_size, device, batch_size=batch_size)
        model_probs_list.append(avg_probs)

    if not model_probs_list:
        raise RuntimeError("No valid models were evaluated!")

    # Ensemble average across all models
    ensemble_probs = np.mean(model_probs_list, axis=0)
    preds = np.argmax(ensemble_probs, axis=1)

    # Build submission DataFrame
    sub_df = pd.DataFrame({
        "id": ids,
        "target": preds
    })

    # Sort strictly by numeric id
    sub_df["id"] = sub_df["id"].astype(int)
    sub_df = sub_df.sort_values("id").reset_index(drop=True)

    output_csv = Path(output_csv)
    sub_df.to_csv(output_csv, index=False)
    print(f"\nSaved submission to: {output_csv}")
    print(f"File size: {output_csv.stat().st_size} bytes")

    # Save probability distribution for post-analysis
    np.save(output_csv.parent / "ensemble_test_probs.npy", ensemble_probs)

    # Verification checks
    print("\n=== Submission Verification ===")
    print(f"Rows count: {len(sub_df)} (Expected: 1070)")
    print(f"Columns: {list(sub_df.columns)} (Expected: ['id', 'target'])")
    print(f"IDs range: {sub_df['id'].min()} to {sub_df['id'].max()} (Unique: {sub_df['id'].nunique()})")
    print(f"Null values:\n{sub_df.isnull().sum()}")
    print("\nPredicted Class Distribution:")
    class_names = [
        "0 - Central Serous Chorioretinopathy",
        "1 - Diabetic Retinopathy",
        "2 - Disc Edema",
        "3 - Glaucoma",
        "4 - Healthy",
        "5 - Macular Scar",
        "6 - Myopia",
        "7 - Pterygium",
        "8 - Retinal Detachment",
        "9 - Retinitis Pigmentosa"
    ]
    dist = sub_df["target"].value_counts().sort_index()
    for c_idx in range(10):
        cnt = dist.get(c_idx, 0)
        pct = (cnt / len(sub_df)) * 100
        print(f"  {class_names[c_idx]}: {cnt} ({pct:.1f}%)")

    print("\nFirst 10 rows:")
    print(sub_df.head(10))
    print("\nLast 10 rows:")
    print(sub_df.tail(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="n:/Ai Dev Fest/submission.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    generate_submission(args.checkpoints, output_csv=args.output, batch_size=args.batch_size)
