# Common imports
import pandas as pd
import hashlib
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# pd.set_option('display.max_columns', 0) # Display any number of columns
# pd.set_option('display.max_rows', 0) # Display any number of rows
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import zipfile
import zlib
import psutil
import sys
import getopt
import time
from datetime import datetime
# import line_profiler
import gc
from contextlib import contextmanager

cwd = os.getcwd()
import sklearn
import seaborn as sns
import multiprocessing
import warnings

warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")  # Ignore useless warnings (see SciPy issue #5998)
from pandas import DataFrame
from sklearn import datasets, metrics
from multiprocessing import Pool
from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer, roc_auc_score
from sklearn.utils import shuffle
from sklearn.ensemble import BaggingClassifier, ExtraTreesRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from scipy.stats import gaussian_kde, anderson, skew, kurtosis, gamma, entropy
from scipy import sparse
from collections import OrderedDict

million = 1000000

# to make this notebook's output stable across runs
# usually when you run and average the results of the same script multiple times (with different seed), the noise tends to be minimized.
# np.random.seed(42)
process = psutil.Process(os.getpid())

# To plot pretty figures
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
np.set_printoptions(linewidth=np.inf)

@contextmanager
def timer(title,Pass=False):
    t0 = time.time()
    print("\n------\t{}......".format(title))
    yield
    print("------\t{} - done in {:.0f}s".format(title, time.time() - t0))

