import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.stdout.reconfigure(line_buffering=True)

class EyeValDataset(Dataset):
    def __init__(self, df, img_size=384):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["proc_path"].values
        self.targets = self.df["target"].values
        self.transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, torch.tensor(self.targets[idx], dtype=torch.long)

@torch.no_grad()
def get_val_probs(ckpt_path, df_val, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    img_size = checkpoint.get("img_size", 384)
    print(f"Loading {model_name}...")

    model = timm.create_model(model_name, pretrained=False, num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    dataset = EyeValDataset(df_val, img_size=img_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_probs = []
    targets_list = []
    for images, targets in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(images)
        probs = torch.softmax(out, dim=1)
        all_probs.append(probs.float().cpu().numpy())
        targets_list.extend(targets.numpy())

    return np.vstack(all_probs), np.array(targets_list)

def main():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Evaluating ensemble on {device}...")

    df = pd.read_csv("n:/Ai Dev Fest/data_384/train_metadata.csv")
    df_val = df[df["fold"] == 0].reset_index(drop=True)

    ckpt1 = "n:/Ai Dev Fest/checkpoints/convnext_small.fb_in22k_ft_in1k_384_fold0_best.pth"
    ckpt2 = "n:/Ai Dev Fest/checkpoints/efficientnet_b3.ra2_in1k_fold0_best.pth"

    probs1, targets = get_val_probs(ckpt1, df_val, device)
    probs2, _ = get_val_probs(ckpt2, df_val, device)

    acc1 = accuracy_score(targets, probs1.argmax(axis=1))
    f1_1 = f1_score(targets, probs1.argmax(axis=1), average='macro')
    print(f"\nModel 1 (ConvNeXt-Small)   - Val Acc: {acc1:.4f} | Macro F1: {f1_1:.4f}")

    acc2 = accuracy_score(targets, probs2.argmax(axis=1))
    f1_2 = f1_score(targets, probs2.argmax(axis=1), average='macro')
    print(f"Model 2 (EfficientNet-B3)  - Val Acc: {acc2:.4f} | Macro F1: {f1_2:.4f}")

    # Ensemble: Weighted blend (0.55 ConvNeXt + 0.45 EfficientNet)
    ensemble_probs = 0.55 * probs1 + 0.45 * probs2
    ens_preds = ensemble_probs.argmax(axis=1)

    ens_acc = accuracy_score(targets, ens_preds)
    ens_f1 = f1_score(targets, ens_preds, average='macro')
    print(f"\n=== ENSEMBLE RESULTS (Fold 0 OOF) ===")
    print(f"Ensemble Accuracy: {ens_acc:.4f} (+{(ens_acc - max(acc1, acc2))*100:.2f}% gain)")
    print(f"Ensemble Macro-F1: {ens_f1:.4f} (+{(ens_f1 - max(f1_1, f1_2))*100:.2f}% gain)")

    class_names = [
        "0 - CSCR",
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
    print("\nClassification Report:")
    print(classification_report(targets, ens_preds, target_names=class_names, digits=4))

if __name__ == "__main__":
    main()
