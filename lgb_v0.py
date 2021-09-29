import logging

import numpy as np
import toad
import pandas as pd
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

    # 特征工程
    ## 找相同的特征
    common_cols = []
    for col in train_public.columns:
        if col in train_internet.columns:
            common_cols.append(col)
    logger.info(len(common_cols))

    ## 只取相同的列
    train_public_less = train_public.loc[:, common_cols]
    train_internet_less = train_internet.loc[:, common_cols]

    # EDA
    is_eda = False
    if is_eda:
        train_public_detect = toad.detect(train_public_less)
        train_internet_detect = toad.detect(train_internet_less)
        train_public_quality = toad.quality(train_public_less, 'isDefault')
        train_internet_quality = toad.quality(train_internet_less, 'isDefault')


    # 处理类别特征
    def class_apply(x):
        if x == 'A':
            return 1
        elif x == 'B':
            return 2
        elif x == 'C':
            return 3
        elif x == 'D':
            return 4
        elif x == 'E':
            return 5
        elif x == 'F':
            return 6
        elif x == 'G':
            return 7
        else:
            return 0


    def employer_type_apply(x):
        if x == '普通企业':
            return 1
        elif x == '幼教与中小学校':
            return 2
        elif x == '高等教育机构':
            return 3
        elif x == '政府机构':
            return 4
        elif x == '上市企业':
            return 5
        elif x == '世界五百强':
            return 6
        else:
            return 0


    def industry_apply(x):
        if x == '金融业':
            return 1
        elif x == '公共服务、社会组织':
            return 2
        elif x == '文化和体育业':
            return 3
        elif x == '信息传输、软件和信息技术服务业':
            return 4
        elif x == '制造业':
            return 5
        elif x == '住宿和餐饮业':
            return 6
        elif x == '建筑业':
            return 7
        elif x == '电力、热力生产供应业':
            return 8
        elif x == '房地产业':
            return 9
        elif x == '交通运输、仓储和邮政业':
            return 10
        elif x == '批发和零售业':
            return 11
        elif x == '农、林、牧、渔业':
            return 12
        elif x == '采矿业':
            return 13
        elif x == '国际组织':
            return 14
        else:
            return 0


    def work_year_apply(x):
        if x == '< 1 year':
            return 1
        elif x == '1 year':
            return 2
        elif x == '2 years':
            return 3
        elif x == '3 years':
            return 4
        elif x == '4 years':
            return 5
        elif x == '5 years':
            return 6
        elif x == '6 years':
            return 7
        elif x == '7 years':
            return 8
        elif x == '8 years':
            return 9
        elif x == '9 years':
            return 10
        elif x == '10+ years':
            return 11
        else:
            return 0


    train_public_less['class'] = train_public_less['class'].apply(class_apply)
    train_public_less['employer_type'] = train_public_less['employer_type'].apply(employer_type_apply)
    train_public_less['industry'] = train_public_less['industry'].apply(industry_apply)
    train_public_less['work_year'] = train_public_less['work_year'].apply(work_year_apply)

    train_internet_less['class'] = train_internet_less['class'].apply(class_apply)
    train_internet_less['employer_type'] = train_internet_less['employer_type'].apply(employer_type_apply)
    train_internet_less['industry'] = train_internet_less['industry'].apply(industry_apply)
    train_internet_less['work_year'] = train_internet_less['work_year'].apply(work_year_apply)

    # 去除非数值类型特征
    drop_cols = list(train_public_less.select_dtypes('object').columns) + ['isDefault']

    # 构造模型数据
    X_train = train_public_less.drop(columns=drop_cols).values
    y_train = train_public_less['isDefault'].values

    X_internet = train_internet_less.drop(columns=drop_cols).values
    y_internet = train_internet_less['isDefault'].values

    # 交叉验证lightgbm
    grid = [{
        'num_leaves': [5, 10, 20, 30],
        'learning_rate': [0.01, 0.03, 0.1, 0.3],
        'reg_alpha': [0, 0.1, 0.2],
        'reg_lambda': [0, 0.1, 0.2]
    }]
    score_detail = []
    best_score = 0
    for param in ParameterGrid(grid):
        logger.info(param)
        param['random_state'] = 1
        scores = []
        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        count = 0
        for train_index, valid_index in skf.split(X_train, y_train):
            count += 1
            logger.info(f'train k fold {count}')
            X_more, X_less = X_train[train_index], X_train[valid_index]
            y_more, y_less = y_train[train_index], y_train[valid_index]
            X_more_internet = np.concatenate([X_more, X_internet])
            y_more_internet = np.concatenate([y_more, y_internet])
            clf = LGBMClassifier(**param)
            clf.fit(X_more_internet, y_more_internet)
            y_proba = clf.predict_proba(X_less)[:, 1]
            score = roc_auc_score(y_less, y_proba)
            scores.append(score)
        scores = np.array(scores)
        logger.info(scores)
        logger.info(scores.mean())
        if scores.mean() > best_score:
            best_score = scores.mean()
            logger.info(f'best score {best_score}')
        score_detail.append([param, scores, scores.mean()])
