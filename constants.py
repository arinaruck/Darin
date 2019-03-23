import numpy as np
import random
import pygame
import time
import cv2
import torch
from torch import utils
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn import datasets


n = 15
goal = 5
field = np.zeros((n, n), dtype=np.int8)

# colors:

black = (0, 0, 0)
white = (255, 255, 255)
marigold = (255, 204, 0)
ciric = (0, 180, 200)

button_1_col = (251, 115, 102)
button_2_col = (69, 0, 252)
button_1_col_pushed = (255, 138, 122)
button_2_col_pushed = (93, 0, 255)

font_col = (27, 97, 160)

colors = {0 : white, 1 : marigold, -1 : ciric}


font_name = None

window_side = 500
WINDOW_SIZE = [window_side, window_side]
MARGIN = WINDOW_SIZE[0] // 100

width = (WINDOW_SIZE[0] - (n + 1) * MARGIN) // n
height = width

b_w, b_h = 135, 60
x_b_1, y_b_1 = 60, window_side - 2 * MARGIN - height - b_h
x_b_2, y_b_2 = window_side - 60 - b_w, window_side - 2 * MARGIN - height - b_h
