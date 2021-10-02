# lgb_v1

## 特征选择
1. 选择共同特征共35个
2. 对class、employer_type、industry、work_year做label编码
3. 去掉了issue_date、earlies_credit_mon两个日期

## 超参数调整
|参数名|参数值|交叉验证auc|
| --- | --- | --- |
|num_leaves|30|0.8744353289285414 ± 0.007191544118111385|