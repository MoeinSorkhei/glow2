"""
Code from this package is taken from: https://github.com/haolsun/dual-glow
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

import os
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .train import *
from .data_io import *
from .init import *



