import logging

import toad
import pandas as pd

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

    # 特征工程
    ## 找相同的特征
    common_cols = []
    for col in train_public.columns:
        if col in train_internet.columns:
            common_cols.append(col)
    logger.info(len(common_cols))

    ## 只取相同的列
    train_public_less = train_public[common_cols]
    train_internet_less = train_internet[common_cols]

    # EDA
    train_public_detect = toad.detect(train_public_less)
    train_internet_detect = toad.detect(train_internet_less)
    train_public_quality = toad.quality(train_public_less, 'isDefault')
    train_internet_quality = toad.quality(train_internet_less, 'isDefault')
