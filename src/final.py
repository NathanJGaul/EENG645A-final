import os
import re
import typing
import datetime
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.models import load_model
from tensorflow.keras.losses import mae
from tensorflow.keras.backend import elu
import tensorflow as tf
import tensorflow_probability as tfp

def main():
    print('Hello World')

if __name__ == "__main__":
    main()
