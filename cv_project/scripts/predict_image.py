"""
Скрипт предсказания класса для любого отдельного изображения:
Сравнивает предсказание Simple CNN и MobileNetV2 с выводом процентов.

Использование:
    python scripts/predict_image.py path/to/image.jpg
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Загрузка моделей...", flush=True)
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ['cup', 'headphones', 'keys', 'pen']

DEVICE = torch.device("cpu")

class SimpleCNN(nn.Module):
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
        return self.classifier(self.features(x))

def load_models():
    b_model = SimpleCNN(len(CLASSES))
    b_path = os.path.join(BASE_DIR, "models", "baseline_cnn.pth")
    if os.path.exists(b_path):
        b_model.load_state_dict(torch.load(b_path, map_location=DEVICE))
    b_model.to(DEVICE).eval()

    imp_model = models.mobilenet_v2(weights=None)
    in_features = imp_model.classifier[1].in_features
    imp_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(CLASSES))
    )
    imp_path = os.path.join(BASE_DIR, "models", "mobilenetv2_improved.pth")
    if os.path.exists(imp_path):
        imp_model.load_state_dict(torch.load(imp_path, map_location=DEVICE))
    imp_model.to(DEVICE).eval()

    return b_model, imp_model

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img_path):
    if not os.path.exists(img_path):
        print(f"[ОШИБКА] Файл не найден: {img_path}", flush=True)
        return

    b_model, imp_model = load_models()
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_b = torch.softmax(b_model(tensor), dim=1)[0]
        out_imp = torch.softmax(imp_model(tensor), dim=1)[0]

    top_b = CLASSES[torch.argmax(out_b).item()]
    top_imp = CLASSES[torch.argmax(out_imp).item()]

    print("\n" + "="*55, flush=True)
    print(f"РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ ДЛЯ: {os.path.basename(img_path)}", flush=True)
    print("="*55, flush=True)
    print(f"1. Baseline Simple CNN       -> {top_b.upper()} ({out_b[torch.argmax(out_b)].item()*100:.1f}%)", flush=True)
    print(f"2. MobileNetV2 (Transfer)     -> {top_imp.upper()} ({out_imp[torch.argmax(out_imp)].item()*100:.1f}%)", flush=True)
    print("-" * 55, flush=True)
    print(f"{'Класс':<15} | {'Baseline CNN':<15} | {'MobileNetV2':<15}", flush=True)
    print("-" * 55, flush=True)
    for i, cls in enumerate(CLASSES):
        print(f"{cls:<15} | {out_b[i].item()*100:6.1f}%          | {out_imp[i].item()*100:6.1f}%", flush=True)
    print("="*55 + "\n", flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        test_cup = os.path.join(BASE_DIR, "dataset", "test", "cup")
        if os.path.exists(test_cup) and len(os.listdir(test_cup)) > 0:
            sample = os.path.join(test_cup, os.listdir(test_cup)[0])
            print(f"[Тест] Запуск на тестовом изображении: {sample}", flush=True)
            predict(sample)
        else:
            print("Укажите путь: python scripts/predict_image.py <путь_к_картинке>", flush=True)
