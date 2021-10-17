# 0.89x

# 导包
from datetime import time
from lightgbm.engine import train
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import seaborn as sns
import gc
import re
import pandas as pd
import lightgbm as lgb
import numpy as np
import toad
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from dateutil.relativedelta import relativedelta
from lightgbm import LGBMClassifier

# 读取数据
train_data = pd.read_csv('../train_public.csv')
test_public = pd.read_csv('../test_public.csv')
train_inte = pd.read_csv('../train_internet.csv')

pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)

train_data, test_public, train_inte = feature_manage(
    train_data, test_public, train_inte)

# 处理两表不同的columns
tr_cols = set(train_data.columns)
same_cols = list(tr_cols.intersection(set(train_inte.columns)))
# train_inteSame是Internet表中的训练数据，取和public相同的列
train_inteSame = train_inte[same_cols].copy()

Inte_add_cos = list(tr_cols.difference(set(same_cols)))
for col in Inte_add_cos:
    train_inteSame[col] = np.nan

# 选择阈值为0.05，从Internet表中提取预测值小于该概率的样本，并对不同来源的样本赋予来源值
y = train_data['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds, IntePre, importances = train_model(
    train_data, train_inteSame, y, folds)

IntePre['isDef'] = train_inte['is_default']  # 标准isDefault
roc_auc_score(IntePre['isDef'], IntePre.isDefault)  # ???
InteId = IntePre.loc[IntePre.isDefault < 0.05, 'loan_id'].tolist()

train_data['dataSource'] = 1
test_public['dataSource'] = 1
train_inteSame['dataSource'] = 0
train_inteSame['isDefault'] = train_inte['is_default']
use_te = train_inteSame[train_inteSame.loan_id.isin(InteId)].copy()
data = pd.concat([train_data, test_public, use_te]).reset_index(drop=True)

# 数据可视化
plt.figure(figsize=(16, 6))
plt.title("Distribution of Default values IntePre")
sns.distplot(IntePre['isDefault'], color="black",
             kde=True, bins=120, label='train_data')
plt.legend()
plt.show()

train = data[data['isDefault'].notna()]
test = data[data['isDefault'].isna()]
del data
del train_data, test_public

y = train['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds, test_preds, importances = train_model(train, test, y, folds)
test_preds.rename({'loan_id': 'id'}, axis=1)[
    ['id', 'isDefault']].to_csv('nn2.csv', index=False)

# 把预测出来概率小于0.5的提取出来重新训练
train_data = pd.read_csv('../train_public.csv')
test_data = pd.read_csv('../test_public.csv')
sub = pd.read_csv('nn2.csv')
sub = sub.rename(columns={'id': 'loan_id'})
sub.loc[sub['isDefault'] < 0.5, 'isDefault'] = 0
nw_sub = sub[(sub['isDefault'] == 0)]
nw_test_data = test_data.merge(nw_sub, on='loan_id', how='inner')
nw_train_data = pd.concat([train_data, nw_test_data]).reset_index(drop=True)
nw_train_data.to_csv('../nw_train_public.csv', index=0)

# 重复训练过程
train_data = pd.read_csv('../nw_train_public.csv')
test_public = pd.read_csv('../submit_example.csv')
train_inte = pd.read_csv('../train_internet.csv')

pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)

train_data, test_public, train_inte = feature_manage(
    train_data, test_public, train_inte)

# 处理两表不同的columns
tr_cols = set(train_data.columns)
same_cols = list(tr_cols.intersection(set(train_inte.columns)))
# train_inteSame是Internet表中的训练数据，取和public相同的列
train_inteSame = train_inte[same_cols].copy()

Inte_add_cos = list(tr_cols.difference(set(same_cols)))
for col in Inte_add_cos:
    train_inteSame[col] = np.nan

y = train_data['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds, IntePre, importances = train_model(
    train_data, train_inteSame, y, folds)

IntePre['isDef'] = train_inte['is_default']
roc_auc_score(IntePre['isDef'], IntePre.isDefault)
InteId = IntePre.loc[IntePre.isDefault < 0.5, 'loan_id'].tolist()

train_data['dataSource'] = 1
test_data['dataSource'] = 1
train_inteSame['dataSource'] = 0
train_inteSame['isDefault'] = train_inte['is_default']
use_te = train_inteSame[train_inteSame.loan_id.isin(InteId)].copy()
data = pd.concat([train_data, test_public, use_te]).reset_index(drop=True)

# 可视化数据
plt.figure(figsize=(16, 6))
plt.title("Distribution of Default values IntePre")
sns.distplot(IntePre['isDefault'], color="black",
             kde=True, bins=120, label='train_data')
plt.legend()
plt.show()
train = data[data['isDefault'].notna()]
test = data[data['isDefault'].isna()]

del data
del train_data, test_public

y = train['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds, test_preds, importances = train_model(train, test, y, folds)
test_preds.rename({'loan_id': 'id'}, axis=1)[
    ['id', 'isDefault']].to_csv('submission.csv', index=False)
