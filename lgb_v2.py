import logging

import numpy as np
import toad
import pandas as pd
import featuretools as ft
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter()
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


if __name__ == '__main__':
    # 读数据
    train_public = pd.read_csv(config.TRAIN_PUBLIC_PATH)
    train_internet = pd.read_csv(config.TRAIN_INTERNET_PATH)
    test_public = pd.read_csv(config.TEST_PUBLIC_PATH)

    # 标签列重命名
    train_internet = train_internet.rename(columns={'is_default': 'isDefault'})

    entities = {
        'train_public': (train_public, )
    }
