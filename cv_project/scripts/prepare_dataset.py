"""
Скрипт валидации, разбиения и структурирования датасета на train/val/test (70% / 15% / 15%).
"""
import os
import shutil
import random
from PIL import Image

CLASSES = ["cup", "keys", "headphones", "pen"]
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def validate_and_split(seed=42):
    random.seed(seed)
    splits = ["train", "val", "test"]
    
    # Создание директорий
    for split in splits:
        for cls in CLASSES:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)
            
    stats = {cls: {"total": 0, "valid": 0, "train": 0, "val": 0, "test": 0} for cls in CLASSES}
    
    for cls in CLASSES:
        cls_raw_dir = os.path.join(RAW_DIR, cls)
        if not os.path.exists(cls_raw_dir):
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Директория {cls_raw_dir} не найдена!")
            continue
            
        files = [f for f in os.listdir(cls_raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        stats[cls]["total"] = len(files)
        
        valid_files = []
        for f in files:
            src = os.path.join(cls_raw_dir, f)
            try:
                with Image.open(src) as img:
                    img.verify()
                with Image.open(src) as img:
                    _ = img.convert("RGB")
                    valid_files.append(f)
            except Exception:
                pass
                
        stats[cls]["valid"] = len(valid_files)
        random.shuffle(valid_files)
        
        n_total = len(valid_files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        
        train_files = valid_files[:n_train]
        val_files = valid_files[n_train:n_train+n_val]
        test_files = valid_files[n_train+n_val:]
        
        stats[cls]["train"] = len(train_files)
        stats[cls]["val"] = len(val_files)
        stats[cls]["test"] = len(test_files)
        
        for f in train_files:
            shutil.copy2(os.path.join(cls_raw_dir, f), os.path.join(DATASET_DIR, "train", cls, f))
        for f in val_files:
            shutil.copy2(os.path.join(cls_raw_dir, f), os.path.join(DATASET_DIR, "val", cls, f))
        for f in test_files:
            shutil.copy2(os.path.join(cls_raw_dir, f), os.path.join(DATASET_DIR, "test", cls, f))

    print("\n================== ИТОГОВАЯ СТАТИСТИКА ДАТАСЕТА ==================")
    print(f"{'Класс':<15} | {'Всего валидных':<15} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
    print("-" * 62)
    tot_train = tot_val = tot_test = tot_valid = 0
    for cls in CLASSES:
        s = stats[cls]
        tot_train += s["train"]
        tot_val += s["val"]
        tot_test += s["test"]
        tot_valid += s["valid"]
        print(f"{cls:<15} | {s['valid']:<15} | {s['train']:<8} | {s['val']:<8} | {s['test']:<8}")
    print("-" * 62)
    print(f"{'ИТОГО':<15} | {tot_valid:<15} | {tot_train:<8} | {tot_val:<8} | {tot_test:<8}")
    print("=================================================================\n")

if __name__ == "__main__":
    validate_and_split()
