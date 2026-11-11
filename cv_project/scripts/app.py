"""
Assignment 2 — Flask Backend: CV Demo с Grad-CAM объяснением
Поддерживает: загрузку изображения → классификация → тепловая карта Grad-CAM
"""

import os
import time
import io
import json
import base64
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from flask import Flask, request, jsonify, send_from_directory

# ─── Константы ─────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
ROOT_DIR    = os.path.dirname(PROJECT_DIR)

# Static dir supports both repo root and cv_project
if os.path.exists(os.path.join(ROOT_DIR, "cv_demo.html")):
    STATIC_DIR = ROOT_DIR
else:
    STATIC_DIR = PROJECT_DIR

if os.path.exists(os.path.join(PROJECT_DIR, "models")):
    MODELS_DIR = os.path.join(PROJECT_DIR, "models")
else:
    MODELS_DIR = os.path.join(ROOT_DIR, "cv_project", "models")
BASE_DIR = ROOT_DIR
CLASSES    = ["cup", "headphones", "keys", "pen"]
CLASS_RU   = {"cup": "Кружка", "headphones": "Наушники", "keys": "Ключи", "pen": "Ручка"}
IMG_SIZE   = 224

def _get_device():
    """
    RTX 5050 Laptop GPU (sm_120) не поддерживается установленным PyTorch cu124.
    Принудительно используем CPU для стабильного и быстрого инференса (~35 ms).
    """
    return torch.device("cpu")

device = _get_device()

# ─── Архитектуры ───────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(128*4*4, 128), nn.ReLU(True),
            nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_mobilenet(num_classes=4):
    m = models.mobilenet_v2(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 256), nn.ReLU(True),
        nn.Dropout(0.2), nn.Linear(256, num_classes)
    )
    return m


# ─── Загрузка моделей ──────────────────────────────────────────────────────────
def load_models():
    baseline = SimpleCNN().to(device)
    improved = get_mobilenet().to(device)

    b_path = os.path.join(MODELS_DIR, "baseline_cnn.pth")
    i_path = os.path.join(MODELS_DIR, "mobilenetv2_improved.pth")

    if os.path.exists(b_path):
        baseline.load_state_dict(torch.load(b_path, map_location=device))
    if os.path.exists(i_path):
        improved.load_state_dict(torch.load(i_path, map_location=device))

    baseline.eval()
    improved.eval()
    return baseline, improved


BASELINE_MODEL, IMPROVED_MODEL = load_models()

# ─── Трансформация ─────────────────────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
    SOTA-метод объяснимости: показывает, какие части изображения
    влияют на предсказание модели.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        # Веса: среднее градиентов по пространственным измерениям
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)   # [1, 1, H, W]
        cam = F.relu(cam)

        # Нормализация в [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Масштабирование до 224×224
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy(), class_idx, F.softmax(output, dim=1)[0].detach().cpu().numpy()


def tensor_to_pil(tensor):
    """Конвертирует нормализованный тензор обратно в PIL Image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.squeeze().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def apply_colormap(cam_np):
    """Применяет цветовую карту jet через OpenCV (без зависимости от версий matplotlib)."""
    import cv2
    cam_uint8 = (np.clip(cam_np, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def overlay_heatmap(original_pil, cam_np, alpha=0.45):
    """Накладывает тепловую карту Grad-CAM на оригинальное изображение."""
    heatmap_rgb = apply_colormap(cam_np)
    heatmap_pil = Image.fromarray(heatmap_rgb).resize(
        (original_pil.width, original_pil.height), Image.BILINEAR
    )
    blended = Image.blend(original_pil.convert("RGB"), heatmap_pil, alpha=alpha)
    return blended


def pil_to_base64(pil_img, fmt="JPEG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/cv_demo.html")
def serve_cv_demo():
    return send_from_directory(STATIC_DIR, "cv_demo.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Принимает: form-data с полем 'image' (файл)
    Возвращает JSON:
    {
        "baseline": { class, confidence, probs, gradcam_b64 },
        "improved":  { class, confidence, probs, gradcam_b64 },
        "original_b64": str
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "Нет поля 'image' в запросе"}), 400

    t0 = time.time()
    file = request.files["image"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Не удалось прочитать изображение: {e}"}), 400

    # Масштабируем до 224×224 для отображения
    display_img = pil_img.resize((IMG_SIZE, IMG_SIZE))

    # Тензор для модели
    inp = val_tf(pil_img).unsqueeze(0).to(device)

    results = {}

    # ── Baseline CNN ────────────────────────────────────────────────────────────
    try:
        # Последний Conv2d в features (index 8)
        target_layer_b = BASELINE_MODEL.features[8]
        gcam_b = GradCAM(BASELINE_MODEL, target_layer_b)
        inp_b = inp.clone().requires_grad_(True)
        cam_b, cls_b, probs_b = gcam_b.generate(inp_b)
        gcam_b.remove_hooks()

        overlay_b = overlay_heatmap(display_img, cam_b)
        results["baseline"] = {
            "class":       CLASSES[cls_b],
            "class_ru":    CLASS_RU[CLASSES[cls_b]],
            "confidence":  float(probs_b[cls_b]),
            "probs":       {CLASSES[i]: float(p) for i, p in enumerate(probs_b)},
            "gradcam_b64": pil_to_base64(overlay_b),
        }
    except Exception as e:
        results["baseline"] = {"error": str(e)}

    # ── MobileNetV2 ─────────────────────────────────────────────────────────────
    try:
        # Последний слой feature extractor
        target_layer_m = IMPROVED_MODEL.features[-1][0]
        gcam_m = GradCAM(IMPROVED_MODEL, target_layer_m)
        inp_m = inp.clone().requires_grad_(True)
        cam_m, cls_m, probs_m = gcam_m.generate(inp_m)
        gcam_m.remove_hooks()

        overlay_m = overlay_heatmap(display_img, cam_m)
        results["improved"] = {
            "class":       CLASSES[cls_m],
            "class_ru":    CLASS_RU[CLASSES[cls_m]],
            "confidence":  float(probs_m[cls_m]),
            "probs":       {CLASSES[i]: float(p) for i, p in enumerate(probs_m)},
            "gradcam_b64": pil_to_base64(overlay_m),
        }
    except Exception as e:
        results["improved"] = {"error": str(e)}

    results["original_b64"] = pil_to_base64(display_img)
    results["latency_ms"] = round((time.time() - t0) * 1000, 1)
    results["device"] = str(device)
    return jsonify(results)


@app.route("/metrics")
def metrics():
    """Возвращает предрассчитанные метрики из metrics_summary.json"""
    path = os.path.join(BASE_DIR, "cv_project", "reports", "metrics_summary.json")
    with open(path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device)})


if __name__ == "__main__":
    print(f"[CV Demo] Запуск на http://localhost:5050")
    print(f"[CV Demo] Устройство: {device}")
    app.run(host="0.0.0.0", port=5050, debug=False)






