"""
Скрипт обучения и сравнения моделей компьютерного зрения:
1. Baseline: Simple Custom CNN
2. Improved: MobileNetV2 (Transfer Learning)

Включает:
- Аугментации данных (Torchvision transforms)
- Метрики: Accuracy, Precision, Recall, F1-score (Macro & Per-class)
- Построение матрицы ошибок и графиков обучения
- Сохранение лучших чекпоинтов моделей
"""

import os
import sys
import json
import time
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Определение устройства (GPU с fallback на CPU)
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    try:
        _ = torch.zeros(1, device="cuda")
        DEVICE = torch.device("cuda:0")
        print(f"[Устройство] Используется GPU: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as e:
        print(f"[Устройство] GPU недоступен для прямых ядер ({e}), используется быстрый многопоточный CPU", flush=True)
else:
    print("[Устройство] Используется CPU", flush=True)

# Гиперпараметры
BATCH_SIZE = 16
IMG_SIZE = 224
EPOCHS_BASELINE = 15
EPOCHS_IMPROVED = 12

# Конвейеры аугментации
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

# ----------------- BASELINE CNN -----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 28x28
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

# ----------------- IMPROVED: MobileNetV2 -----------------
def get_mobilenet_model(num_classes=4):
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

# ----------------- МЕТРИКИ И АНАЛИЗ -----------------
def compute_metrics(y_true, y_pred, class_names):
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total_samples = len(y_true)
    correct = np.sum([cm[i, i] for i in range(n_classes)])
    accuracy = correct / total_samples if total_samples > 0 else 0.0

    report = {"accuracy": float(accuracy)}
    precisions, recalls, f1s = [], [], []

    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        report[cls] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1-score": float(f1),
            "support": int(np.sum(cm[i, :]))
        }

    report["macro avg"] = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1-score": float(np.mean(f1s))
    }
    return report, cm

# ----------------- ОБУЧЕНИЕ -----------------
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"  Epoch {epoch + 1:02d}/{num_epochs:02d} | ", end="", flush=True)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f"{phase.upper()}: Loss={epoch_loss:.3f}, Acc={epoch_acc*100:.1f}% | ", end="", flush=True)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f"  --> Обучение завершено за {time_elapsed // 60:.0f}м {time_elapsed % 60:.0f}с. Лучшая Val Acc: {best_acc*100:.2f}%\n", flush=True)
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    return compute_metrics(all_labels, all_preds, class_names)

def plot_learning_curves(history, model_name, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    ax1.set_title(f'{model_name}: Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax1_acc = ax2
    ax1_acc.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Acc (%)')
    ax1_acc.plot(epochs, [a * 100 for a in history['val_acc']], 'r-s', label='Val Acc (%)')
    ax1_acc.set_title(f'{model_name}: Accuracy Curve')
    ax1_acc.set_xlabel('Epoch')
    ax1_acc.set_ylabel('Accuracy (%)')
    ax1_acc.grid(True, alpha=0.3)
    ax1_acc.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, class_names, model_name, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_yticklabels(class_names)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10, weight='bold')

    plt.title(f"Confusion Matrix: {model_name}", pad=20)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    print("=== ЗАПУСК ПАЙПЛАЙНА ОБУЧЕНИЯ МОДЕЛЕЙ ===", flush=True)
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=0)
        for x in ['train', 'val', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Классы: {class_names}", flush=True)
    print(f"Разбиение данных: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}, Test={dataset_sizes['test']}\n", flush=True)

    results = {}

    # 1. Baseline CNN
    print("--------------------------------------------------", flush=True)
    print("1. Обучение Базовой Модели (Simple Custom CNN)...", flush=True)
    print("--------------------------------------------------", flush=True)
    baseline_model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer_b = optim.Adam(baseline_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_b = optim.lr_scheduler.StepLR(optimizer_b, step_size=5, gamma=0.5)

    baseline_model, hist_b = train_model(
        baseline_model, criterion, optimizer_b, scheduler_b,
        dataloaders, dataset_sizes, num_epochs=EPOCHS_BASELINE
    )
    torch.save(baseline_model.state_dict(), os.path.join(MODELS_DIR, "baseline_cnn.pth"))
    plot_learning_curves(hist_b, "Simple CNN (Baseline)", os.path.join(FIGURES_DIR, "loss_acc_baseline.png"))
    rep_b, cm_b = evaluate_model(baseline_model, dataloaders['test'], class_names)
    plot_confusion_matrix(cm_b, class_names, "Simple CNN (Baseline)", os.path.join(FIGURES_DIR, "cm_baseline.png"))
    results["baseline_cnn"] = {"report": rep_b, "history": hist_b}

    # 2. Improved Model
    print("--------------------------------------------------", flush=True)
    print("2. Обучение Улучшенной Модели (MobileNetV2 Transfer Learning)...", flush=True)
    print("--------------------------------------------------", flush=True)
    improved_model = get_mobilenet_model(num_classes=len(class_names)).to(DEVICE)
    optimizer_imp = optim.Adam([
        {'params': improved_model.features.parameters(), 'lr': 1e-4},
        {'params': improved_model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    scheduler_imp = optim.lr_scheduler.CosineAnnealingLR(optimizer_imp, T_max=EPOCHS_IMPROVED, eta_min=1e-6)

    improved_model, hist_imp = train_model(
        improved_model, criterion, optimizer_imp, scheduler_imp,
        dataloaders, dataset_sizes, num_epochs=EPOCHS_IMPROVED
    )
    torch.save(improved_model.state_dict(), os.path.join(MODELS_DIR, "mobilenetv2_improved.pth"))
    plot_learning_curves(hist_imp, "MobileNetV2 (Transfer Learning)", os.path.join(FIGURES_DIR, "loss_acc_improved.png"))
    rep_imp, cm_imp = evaluate_model(improved_model, dataloaders['test'], class_names)
    plot_confusion_matrix(cm_imp, class_names, "MobileNetV2 (Transfer Learning)", os.path.join(FIGURES_DIR, "cm_improved.png"))
    results["mobilenetv2"] = {"report": rep_imp, "history": hist_imp}

    # Сохранение всех результатов
    with open(os.path.join(REPORTS_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("================================================================================", flush=True)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ (НА ТЕСТОВОЙ ВЫБОРКЕ):", flush=True)
    print("================================================================================", flush=True)
    print(f"{'Модель':<25} | {'Accuracy':<10} | {'Precision (Macro)':<18} | {'Recall (Macro)':<15} | {'F1-Score (Macro)':<15}", flush=True)
    print("-" * 88, flush=True)
    for m_name, label in [("baseline_cnn", "Simple Custom CNN"), ("mobilenetv2", "MobileNetV2 Transfer")]:
        r = results[m_name]["report"]
        acc = r["accuracy"]
        prec = r["macro avg"]["precision"]
        rec = r["macro avg"]["recall"]
        f1 = r["macro avg"]["f1-score"]
        print(f"{label:<25} | {acc*100:6.2f}%    | {prec*100:14.2f}%    | {rec*100:11.2f}%    | {f1*100:11.2f}%", flush=True)
    print("================================================================================\n", flush=True)

if __name__ == "__main__":
    main()
