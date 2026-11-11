import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch
import torchvision
from torchvision import transforms, models

print(f"NumPy: {np.__version__}", flush=True)
print(f"PIL: {PIL.__version__}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"Torchvision: {torchvision.__version__}", flush=True)
print("SUCCESS: Core vision pipeline is ready!", flush=True)
