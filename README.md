# lgb_v1

## 特征选择
1. 选择共同特征共35个
2. 对class、employer_type、industry、work_year做label编码
3. 去掉了issue_date、earlies_credit_mon两个日期

## 超参数调整
|参数名|参数值|交叉验证auc|
| --- | --- | --- |
|num_leaves|30|0.8744353289285414 ± 0.007191544118111385|



# feature

## 特征工程v1

1. 填补所有表的缺失值
2. industry和employer_type换成了one-hot编码（但每一个职业都变成了一个特征 感觉特征稍微有点多了）
3. issue_data取了[年份]，[月份]和[星期]这3个新特征

## 超参数调整

| num_leaves | max_depth | n_estimators | 线下交叉验证auc    | 线上auc       |
| ---------- | --------- | ------------ | ------------------ | ------------- |
| 32         | 6         | 100          | 0.8774697634061044 | 0.85259483436 |

（参数参考了网上的一个LGBMClassifier）