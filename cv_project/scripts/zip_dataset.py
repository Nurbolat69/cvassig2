"""
Скрипт для архивации датасета в dataset.zip согласно требованиям задания:
"Dataset (organized and zipped or shared via Google Drive)"
"""

import os
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ZIP_OUTPUT_PATH = os.path.join(BASE_DIR, "dataset.zip")

def create_dataset_zip():
    if not os.path.exists(DATASET_DIR):
        print(f"[ОШИБКА] Директория {DATASET_DIR} не существует.")
        return

    print(f"Архивация датасета из {DATASET_DIR} в {ZIP_OUTPUT_PATH}...")
    file_count = 0
    with zipfile.ZipFile(ZIP_OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(DATASET_DIR):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, BASE_DIR)
                    zipf.write(file_path, arcname)
                    file_count += 1

    size_mb = os.path.getsize(ZIP_OUTPUT_PATH) / (1024 * 1024)
    print(f"Успешно упаковано {file_count} изображений в {ZIP_OUTPUT_PATH} ({size_mb:.2f} MB).")

if __name__ == "__main__":
    create_dataset_zip()
