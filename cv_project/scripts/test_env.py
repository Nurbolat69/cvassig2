import os, sys
print("A: os, sys imported", flush=True)

import numpy as np
print(f"B: numpy {np.__version__}", flush=True)

import matplotlib
matplotlib.use('Agg') # headless backend
import matplotlib.pyplot as plt
print("C: matplotlib OK", flush=True)

import sklearn
print(f"D: sklearn {sklearn.__version__}", flush=True)

import PIL
print(f"E: PIL {PIL.__version__}", flush=True)

import torch
print(f"F: torch {torch.__version__}", flush=True)

import torchvision
print("G: torchvision OK", flush=True)

print("H: All core modules loaded cleanly!", flush=True)
