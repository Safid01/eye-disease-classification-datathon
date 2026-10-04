import os
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

RAW_TRAIN_DIR = Path("n:/Ai Dev Fest/train")
RAW_TEST_DIR = Path("n:/Ai Dev Fest/test/all_test")

PROC_DIR = Path("n:/Ai Dev Fest/data_384")
PROC_TRAIN_DIR = PROC_DIR / "train"
PROC_TEST_DIR = PROC_DIR / "test"

PROC_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
PROC_TEST_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAPPING = {
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

TARGET_SIZE = (384, 384)

def process_single_image(args):
    src_path, dst_path = args
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return True
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return False
        # Resize using INTER_AREA for downsampling
        resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def main():
    print("Collecting train images...")
    train_records = []
    tasks = []
    
    for folder in RAW_TRAIN_DIR.iterdir():
        if folder.is_dir() and folder.name in CLASS_MAPPING:
            label = CLASS_MAPPING[folder.name]
            files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))
            for f in files:
                dst_name = f"{label}_{f.name}"
                dst_path = PROC_TRAIN_DIR / dst_name
                train_records.append({
                    "raw_path": str(f),
                    "proc_path": str(dst_path),
                    "file_name": dst_name,
                    "class_name": folder.name,
                    "target": label
                })
                tasks.append((f, dst_path))
    
    df_train = pd.DataFrame(train_records)
    print(f"Total train images: {len(df_train)}")
    
    print("Collecting test images...")
    test_records = []
    test_files = sorted(list(RAW_TEST_DIR.glob("*.jpg")) + list(RAW_TEST_DIR.glob("*.png")), key=lambda p: int(p.stem))
    for f in test_files:
        test_id = int(f.stem)
        dst_name = f.name
        dst_path = PROC_TEST_DIR / dst_name
        test_records.append({
            "id": test_id,
            "raw_path": str(f),
            "proc_path": str(dst_path),
            "file_name": dst_name
        })
        tasks.append((f, dst_path))
        
    df_test = pd.DataFrame(test_records)
    print(f"Total test images: {len(df_test)}")
    
    # Stratified 5-Fold Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_train["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df_train, df_train["target"])):
        df_train.loc[val_idx, "fold"] = fold_idx
        
    df_train.to_csv(PROC_DIR / "train_metadata.csv", index=False)
    df_test.to_csv(PROC_DIR / "test_metadata.csv", index=False)
    print(f"Metadata saved to {PROC_DIR}!")
    
    print(f"Starting parallel preprocessing of {len(tasks)} images to 384x384...")
    # Use max CPU workers
    workers = min(16, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="Resizing images"))
    
    success = sum(results)
    print(f"Successfully preprocessed {success}/{len(tasks)} images!")

if __name__ == "__main__":
    main()
