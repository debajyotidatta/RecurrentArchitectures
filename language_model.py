"""
Imports
"""
import numpy as np
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import time
import os
from tensorflow.models.rnn.ptb import reader

pwd
with open('./data/tiny-shakespeare.txt','r') as f:
    raw_data = f.read()
    print("Data length:", len(raw_data))
