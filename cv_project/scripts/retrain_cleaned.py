# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Retrain models on CLEANED dataset - Assignment 2
Baseline CNN + MobileNetV2 on the improved dataset
"""

import os, json, copy, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ───────────────────────────────────────────────────────────────────
SEED       = 42
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS_B   = 20   # Baseline - больше эпох на чистых данных
EPOCHS_M   = 15   # MobileNetV2

BASE       = r'C:\Users\nurbo\Projects\spandiyar-kobeev-journey\cv_project'
DATA_DIR   = os.path.join(BASE, 'dataset')
MODELS_DIR = os.path.join(BASE, 'models')
REPORTS_DIR = os.path.join(BASE, 'reports')
FIGS_DIR   = os.path.join(REPORTS_DIR, 'figures')

os.makedirs(FIGS_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)

# CPU (RTX 5050 not compatible with this PyTorch build)
device = torch.device('cpu')
print('Device: ' + str(device))
print('Note: Using CPU (RTX 5050 requires PyTorch with sm_120 support)')

# ── Data ─────────────────────────────────────────────────────────────────────
data_tf = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),              # увеличили с 15 до 20
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_tf[x])
                  for x in ['train', 'val', 'test']}
loaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                         shuffle=(x=='train'), num_workers=0)
           for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
classes = image_datasets['train'].classes
print('Classes: ' + str(classes))
print('Sizes: ' + str(dataset_sizes))

# ── Architectures ─────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(128*4*4, 128), nn.ReLU(True),
            nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


def get_mobilenet(num_classes=4):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 256), nn.ReLU(True),
        nn.Dropout(0.2), nn.Linear(256, num_classes)
    )
    return m

# ── Train loop ────────────────────────────────────────────────────────────────
def train_model(model, optimizer, scheduler, num_epochs, model_name):
    criterion = nn.CrossEntropyLoss()
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    print('\n--- Training ' + model_name + ' for ' + str(num_epochs) + ' epochs ---')
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = running_corrects = 0

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            if scheduler and phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects / dataset_sizes[phase]
            history[phase + '_loss'].append(epoch_loss)
            history[phase + '_acc'].append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())

        print('Epoch {:2d}/{} | train_loss={:.4f} train_acc={:.4f} | val_acc={:.4f}'.format(
            epoch+1, num_epochs,
            history['train_loss'][-1], history['train_acc'][-1], history['val_acc'][-1]))

    model.load_state_dict(best_wts)
    print(model_name + ' best val acc: ' + str(round(best_acc, 4)))
    return model, history


def evaluate(model, model_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds,
                                   target_names=classes, output_dict=True)
    cm     = confusion_matrix(all_labels, all_preds)

    print('\n--- ' + model_name + ' TEST RESULTS ---')
    print(classification_report(all_labels, all_preds, target_names=classes))

    return report, cm


def save_training_plot(history, model_name, suffix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(model_name + ' - Training Curves (Cleaned Dataset)', fontsize=13)
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val Loss', markersize=4)
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=4)
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val Acc', markersize=4)
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'loss_acc_' + suffix + '_v2.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: ' + out)


def save_cm_plot(cm, model_name, suffix):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(model_name + ' - Confusion Matrix (Cleaned Dataset)')
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'cm_' + suffix + '_v2.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: ' + out)


# ═══════════════════════════════════════════════════════════════════
#  BASELINE CNN
# ═══════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  BASELINE CNN (Cleaned Dataset)')
print('='*60)
baseline = SimpleCNN(len(classes)).to(device)
opt_b = optim.Adam(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
sch_b = optim.lr_scheduler.StepLR(opt_b, step_size=5, gamma=0.5)
baseline, hist_b = train_model(baseline, opt_b, sch_b, EPOCHS_B, 'SimpleCNN')

torch.save(baseline.state_dict(), os.path.join(MODELS_DIR, 'baseline_cnn.pth'))
print('Saved: baseline_cnn.pth')
save_training_plot(hist_b, 'SimpleCNN', 'baseline')
report_b, cm_b = evaluate(baseline, 'SimpleCNN')
save_cm_plot(cm_b, 'SimpleCNN', 'baseline')

# ═══════════════════════════════════════════════════════════════════
#  MOBILENETV2
# ═══════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  MOBILENETV2 (Cleaned Dataset + Transfer Learning)')
print('='*60)
improved = get_mobilenet(len(classes)).to(device)

# Differential learning rates
backbone_params = [p for n,p in improved.named_parameters() if 'classifier' not in n]
head_params     = list(improved.classifier.parameters())
opt_m = optim.Adam([
    {'params': backbone_params, 'lr': 1e-4},
    {'params': head_params,     'lr': 1e-3},
], weight_decay=1e-4)
sch_m = optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=EPOCHS_M)
improved, hist_m = train_model(improved, opt_m, sch_m, EPOCHS_M, 'MobileNetV2')

torch.save(improved.state_dict(), os.path.join(MODELS_DIR, 'mobilenetv2_improved.pth'))
print('Saved: mobilenetv2_improved.pth')
save_training_plot(hist_m, 'MobileNetV2', 'improved')
report_m, cm_m = evaluate(improved, 'MobileNetV2')
save_cm_plot(cm_m, 'MobileNetV2', 'improved')

# ═══════════════════════════════════════════════════════════════════
#  SAVE METRICS SUMMARY
# ═══════════════════════════════════════════════════════════════════
summary = {
    'baseline_cnn': {'report': report_b, 'history': hist_b},
    'mobilenetv2':  {'report': report_m, 'history': hist_m},
}
out_json = os.path.join(REPORTS_DIR, 'metrics_summary.json')
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=4)
print('\nSaved metrics: ' + out_json)

print('\n' + '='*60)
print('  COMPARISON: A1 (dirty) vs A2 (cleaned)')
print('='*60)
print('Baseline CNN:')
print('  A1 accuracy: 41.67%  -->  A2 accuracy: ' + str(round(report_b['accuracy']*100, 2)) + '%')
print('MobileNetV2:')
print('  A1 accuracy: 70.83%  -->  A2 accuracy: ' + str(round(report_m['accuracy']*100, 2)) + '%')
print('\nDONE! Models retrained on cleaned dataset.')
