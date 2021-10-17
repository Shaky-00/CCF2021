from lightgbm.engine import train
from numpy.core.defchararray import find


def workYearDic(x):
    if(str(x) == 'nan'):
        return -1
    x = x.replace('<1', '0')
    return int(re.search('(\d+)', x).group())


def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'

# 特征工程
# 一些基本的字典映射和时间特征处理


def feature_manage(train_data, test_public, train_inte):
    timeMax = pd.to_datetime('1-Dec-21')
    class_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

    train_data['work_year'] = train_data['work_year'].map(workYearDic)
    test_public['work_year'] = test_public['work_year'].map(workYearDic)
    train_inte['work_year'] = train_inte['work_year'].map(workYearDic)

    train_data['class'] = train_data['class'].map(class_dict)
    test_public['class'] = test_public['class'].map(class_dict)
    train_inte['class'] = train_inte['class'].map(class_dict)

    train_data['earlies_credit_mon'] = pd.to_datetime(
        train_data['earlies_credit_mon'].map(findDig))
    test_public['earlies_credit_mon'] = pd.to_datetime(
        test_public['earlies_credit_mon'].map(findDig))
    train_inte['earlies_credit_mon'] = pd.to_datetime(
        train_inte['earlies_credit_mon'].map(findDig))

    train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = train_data.loc[train_data['earlies_credit_mon']
                                                                                                      > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(years=-100)
    test_public.loc[test_public['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = test_public.loc[test_public['earlies_credit_mon']
                                                                                                         > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(years=-100)
    train_inte.loc[train_inte['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = train_inte.loc[train_inte['earlies_credit_mon']
                                                                                                      > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(years=-100)
    train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
    test_public['issue_date'] = pd.to_datetime(test_public['issue_date'])
    train_inte['issue_date'] = pd.to_datetime(train_inte['issue_date'])

    train_data['issue_date_month'] = train_data['issue_date'].dt.month
    test_public['issue_date_month'] = test_public['issue_date'].dt.month
    train_inte['issue_date_month'] = train_inte['issue_date'].dt.month

    train_data['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek
    test_public['issue_date_dayofweek'] = test_public['issue_date'].dt.dayofweek
    train_inte['issue_date_dayofweek'] = train_inte['issue_date'].dt.dayofweek

    train_data['earliesCreditMon'] = train_data['earlies_credit_mon'].dt.month
    test_public['earliesCreditMon'] = test_public['earlies_credit_mon'].dt.month
    train_inte['earliesCreditMon'] = train_inte['earlies_credit_mon'].dt.month

    train_data['earliesCreditYear'] = train_data['earlies_credit_mon'].dt.year
    test_public['earliesCreditYear'] = test_public['earlies_credit_mon'].dt.year
    train_inte['earliesCreditYear'] = train_inte['earlies_credit_mon'].dt.year

    # employer_type和industry的编码
    cat_cols = ['employer_type', 'industry']
    for col in cat_cols:
        lbl = LableEncoder().fit(train_data[col])
        train_data[col] = lbl.transform(train_data[col])
        test_public[col] = lbl.transform(test_public[col])
        train_inte[col] = lbl.transform(train_inte[col])

    # 去掉可能影响最终结果的特征以及之前处理过的特征
    col_to_drop = ['issue_date', 'earlies_credit_mon']
    train_data = train_data.drop(col_to_drop, axis=1)
    test_public = test_public.drop(col_to_drop, axis=1)
    train_inte = train_inte.drop(col_to_drop, axis=1)

    # 使用toad进行分箱
    c = toad.transform.Combiner()
    chosen_cols = ['total_loan', 'interest', 'monthly_payment', 'debt_loan_ratio', 'scoring_low', 'scoring_high',
                   'recircle_b', 'recircle_u', 'early_return_amount', 'early_return_amount_3mon',
                   'isDefault']
    box_train = train_data[train_data['isDefault'].notna()]
    c.fit(box_train[chosen_cols], y='isDefault', method='dt', min_samples=0.05)

    train_data = c.transform(train_data)
    test_public = c.transform(test_public)
    train_inte = c.transform(train_inte)

    return train_data, test_public, train_inte
