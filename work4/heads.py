import sys
import os

import math

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms as T

from matplotlib import pyplot as plt
from matplotlib import rcParams

import numpy as np

from torch.distributions import Normal, kl_divergence
import random