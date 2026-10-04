import os
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

train_dir = Path("n:/Ai Dev Fest/train")
test_dir = Path("n:/Ai Dev Fest/test/all_test")

class_mapping = {
    "Central Serous Chorioretinopathy [Color Fundus]": 0,
    "Diabetic Retinopathy": 1,
    "Disc Edema": 2,
    "Glaucoma": 3,
    "Healthy": 4,
    "Macular Scar": 5,
    "Myopia": 6,
    "Pterygium": 7,
    "Retinal Detachment": 8,
    "Retinitis Pigmentosa": 9
}

print("=== Class Mapping & Counts ===")
train_records = []
for folder in train_dir.iterdir():
    if folder.is_dir():
        cname = folder.name
        if cname in class_mapping:
            label = class_mapping[cname]
            files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
            print(f"Class {label}: '{cname}' -> {len(files)} images")
            for f in files:
                train_records.append({"path": str(f), "class_name": cname, "label": label})
        else:
            print(f"UNKNOWN FOLDER: {cname}")

df_train = pd.DataFrame(train_records)
print(f"\nTotal train samples: {len(df_train)}")
print("Train class distribution:")
print(df_train['label'].value_counts().sort_index())

test_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg"))
print(f"\nTotal test samples: {len(test_files)}")

# Check image dimensions on a random sample
sample_sizes = []
for p in df_train['path'].sample(min(50, len(df_train)), random_state=42):
    with Image.open(p) as img:
        sample_sizes.append(img.size)

sizes_arr = np.array(sample_sizes)
print("\nSample image sizes (Width, Height):")
print(f"Min: {sizes_arr.min(axis=0)}, Max: {sizes_arr.max(axis=0)}, Mean: {sizes_arr.mean(axis=0).round(1)}")

# Check test IDs
test_ids = []
for tf in test_files:
    test_ids.append(int(tf.stem))
test_ids.sort()
print(f"\nTest IDs range: min={min(test_ids)}, max={max(test_ids)}, count={len(test_ids)}, unique={len(set(test_ids))}")
print(f"First 10 test IDs: {test_ids[:10]}")
print(f"Last 10 test IDs: {test_ids[-10:]}")
