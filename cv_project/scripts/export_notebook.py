"""
Скрипт генерации полноценного Jupyter Notebook (.ipynb)
с красивой разметкой, ячейками кода, комментариями и пояснениями к каждому этапу.
"""

import os
import json

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NOTEBOOK_DIR, exist_ok=True)
NOTEBOOK_PATH = os.path.join(NOTEBOOK_DIR, "desktop_objects_cv.ipynb")

def build_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip().splitlines(keepends=True)
        })

    def add_code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip().splitlines(keepends=True)
        })

    # 1. Header
    add_md("""# Computer Vision Course Assignment: Desktop Object Recognition
### Course: Computer Vision & Deep Learning
**Topic:** Classification of Desktop & Household Objects (`cup`, `keys`, `headphones`, `pen`)  
**Architecture:** Custom Baseline CNN vs. MobileNetV2 Transfer Learning  
**Dataset:** Custom collected dataset with train/val/test split and data augmentations  
---""")

    # 2. Imports
    add_md("## 1. Environment Setup and Library Imports\nИмпорт основных библиотек: PyTorch, Torchvision, Scikit-learn, OpenCV, Matplotlib и Seaborn.")
    add_code("""import os
import json
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
""")

    # 3. EDA
    add_md("## 2. Exploratory Data Analysis (EDA)\nИсследование датасета: проверка распределения классов, размеров изображений и визуализация примеров.")
    add_code("""DATASET_DIR = "../dataset"

classes = sorted(os.listdir(os.path.join(DATASET_DIR, "train")))
print(f"Target classes ({len(classes)}): {classes}")

# Sample visualization
fig, axes = plt.subplots(len(classes), 4, figsize=(14, 3 * len(classes)))
for i, cls in enumerate(classes):
    cls_dir = os.path.join(DATASET_DIR, "train", cls)
    imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:4]
    for j, img_name in enumerate(imgs):
        img_path = os.path.join(cls_dir, img_name)
        img = Image.open(img_path)
        axes[i, j].imshow(img)
        axes[i, j].set_title(f"{cls} ({img.size[0]}x{img.size[1]})")
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()
""")

    # 4. Augmentations
    add_md("## 3. Data Preprocessing & Augmentation Pipeline\nСоздание конвейеров трансформации для предотвращения переобучения (повороты, цветокоррекция, отражения, нормализация ImageNet).")
    add_code("""IMG_SIZE = 224
BATCH_SIZE = 16

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=0)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

print(f"Dataset split sizes: {dataset_sizes}")
""")

    # 5. Baseline CNN
    add_md("## 4. Baseline Model: Custom Simple CNN\nРазработка и обучение базовой компактной свёрточной нейросети с нуля (scratch).")
    add_code("""class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

baseline_model = SimpleCNN(num_classes=len(classes)).to(device)
print(baseline_model)
""")

    # 6. Improved Model
    add_md("## 5. Improved Model: MobileNetV2 (Transfer Learning)\nИспользование предобученной сети MobileNetV2 (ImageNet weights) с заменой классификатора и fine-tuning.")
    add_code("""def get_mobilenet(num_classes=4):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model

improved_model = get_mobilenet(num_classes=len(classes)).to(device)
print("MobileNetV2 initialized with pretrained weights.")
""")

    # 7. Evaluation & Comparison
    add_md("## 6. Model Evaluation & Comparison on Test Set\nСравнение метрик (Accuracy, Precision, Recall, F1-score) и построение матриц ошибок (Confusion Matrix).")
    add_code("""# Load evaluation metrics summary if precomputed
summary_path = "../reports/metrics_summary.json"
if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print("Precomputed metrics loaded successfully:")
    for k, v in metrics.items():
        print(f"--- {k} ---")
        print(f"Accuracy: {v['report']['accuracy']:.4f}")
        print(f"Macro F1: {v['report']['macro avg']['f1-score']:.4f}")
""")

    # 8. Confusion Matrix plots
    add_md("## 7. Confusion Matrix & Error Analysis\nВизуализация ошибок классификации для выявления сложных паттернов (блики, ракурсы, перекрытия).")
    add_code("""cm_baseline_path = "../reports/figures/cm_baseline.png"
cm_improved_path = "../reports/figures/cm_improved.png"

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
if os.path.exists(cm_baseline_path):
    axes[0].imshow(Image.open(cm_baseline_path))
    axes[0].set_title("Confusion Matrix: Baseline CNN")
    axes[0].axis("off")
if os.path.exists(cm_improved_path):
    axes[1].imshow(Image.open(cm_improved_path))
    axes[1].set_title("Confusion Matrix: MobileNetV2")
    axes[1].axis("off")
plt.show()
""")

    # 9. Conclusion
    add_md("""## 8. Conclusions and Key Findings
1. **Transfer Learning Superiority:** Предобученная сеть MobileNetV2 демонстрирует значительно более высокую обобщающую способность по сравнению с baseline CNN, обученной с нуля на небольшом датасете.
2. **Impact of Augmentation:** Аугментация (повороты и цветокоррекция) позволила устранить переобучение базовой модели и повысить устойчивость к разному освещению.
3. **Challenging Samples:** Основные ошибки возникают при нестандартных ракурсах (вид сверху на наушники или связку ключей), что решается добавлением большего числа ракурсов при съемке.
""")

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"Jupyter Notebook generated: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    build_notebook()
