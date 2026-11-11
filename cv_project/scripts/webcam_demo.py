"""
Демонстрация в реальном времени с веб-камеры:
Запускает обученную модель MobileNetV2 Transfer Learning и классифицирует
предметы перед камерой (cup, keys, headphones, pen) в реальном времени.

Запуск:
    python scripts/webcam_demo.py
Выход: клавиша 'Q'
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv2_improved.pth")

CLASSES = ['cup', 'headphones', 'keys', 'pen']
RU_CLASSES = {
    'cup': 'Кружка (Cup)',
    'headphones': 'Наушники (Headphones)',
    'keys': 'Ключи (Keys)',
    'pen': 'Ручка (Pen)'
}

DEVICE = torch.device("cpu")

def load_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(CLASSES))
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"[OK] Модель успешно загружена: {MODEL_PATH}")
    else:
        print(f"[ОШИБКА] Файл весов {MODEL_PATH} не найден!")
        sys.exit(1)
        
    model.to(DEVICE)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ОШИБКА] Веб-камера не найдена или недоступна.")
        return

    print("\n" + "="*50)
    print(">>> ДЕМО РАСПОЗНАВАНИЯ ПРЕДМЕТОВ С ВЕБ-КАМЕРЫ")
    print("Направьте предмет (кружка, ключи, наушники, ручка) в камеру.")
    print("Для выхода нажмите клавишу 'Q'.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.topk(probs, 1)

        pred_class = CLASSES[top_idx.item()]
        pred_label = RU_CLASSES.get(pred_class, pred_class)
        conf = top_prob.item() * 100

        h, w, _ = frame.shape
        cv2.rectangle(frame, (10, 10), (w - 10, 100), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 100), (0, 200, 255), 2)

        color = (0, 255, 0) if conf > 60 else (0, 165, 255)
        text_pred = f"Pred: {pred_label}"
        text_conf = f"Confidence: {conf:.1f}%"

        cv2.putText(frame, text_pred, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, text_conf, (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Вывод мини-вероятностей всех классов снизу
        y_offset = h - 100
        cv2.rectangle(frame, (10, y_offset - 10), (320, h - 10), (20, 20, 20), -1)
        for i, cls in enumerate(CLASSES):
            p = probs[i].item() * 100
            bar_text = f"{cls:<10}: {p:4.1f}%"
            cv2.putText(frame, bar_text, (20, y_offset + i * 20 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Desktop Object Recognition - Live Demo (Press Q to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
