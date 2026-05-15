import os
import sys
import cv2
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch):", torch.version.cuda)

if torch.cuda.is_available():
  print("GPU:", torch.cuda.get_device_name(0))