#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pandas as pd
warnings.filterwarnings('ignore')

train_bank = pd.read_csv('./train_public.csv')
train_internet = pd.read_csv('./train_internet.csv')
train_bank = train_bank.rename(columns={'isDefault': 'is_default'})


# In[2]:


test = pd.read_csv('./test_public.csv')


# ### 数据预处理

# In[3]:


common_cols = []
for col in train_bank.columns:
    if col in train_internet.columns:
        common_cols.append(col)
    else: continue
len(common_cols)


# In[4]:


print(len(train_bank.columns))
print(len(train_internet.columns))


# In[5]:


train_bank_left = list(set(list(train_bank.columns)) - set(common_cols))
train_internet_left = list(set(list(train_internet.columns)) - set(common_cols))

train_bank_left


# In[6]:


train_internet_left


# In[7]:


train1_data = train_internet[common_cols]
train2_data = train_bank[common_cols]
test_data = test[common_cols[:-1]]


# In[8]:


import datetime

# 日期类型：issueDate，earliesCreditLine
# 转换为pandas中的日期类型
train1_data['issue_date'] = pd.to_datetime(train1_data['issue_date'])
# 提取多尺度特征
train1_data['issue_date_y'] = train1_data['issue_date'].dt.year
train1_data['issue_date_m'] = train1_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
train1_data['issue_date_diff'] = train1_data['issue_date'].apply(lambda x: x-base_time).dt.days
train1_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
train1_data.drop('issue_date', axis = 1, inplace = True)


# In[9]:


# 日期类型：issueDate，earliesCreditLine
# 转换为pandas中的日期类型
train2_data['issue_date'] = pd.to_datetime(train2_data['issue_date'])
# 提取多尺度特征
train2_data['issue_date_y'] = train2_data['issue_date'].dt.year
train2_data['issue_date_m'] = train2_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
train2_data['issue_date_diff'] = train2_data['issue_date'].apply(lambda x: x-base_time).dt.days
train2_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
train2_data.drop('issue_date', axis = 1, inplace = True)
train2_data


# In[10]:


employer_type = train1_data['employer_type'].value_counts().index
industry = train1_data['industry'].value_counts().index


# In[11]:


emp_type_dict = dict(zip(employer_type, [0,1,2,3,4,5]))
industry_dict = dict(zip(industry, [i for i in range(15)]))


# In[12]:


train1_data['work_year'].fillna('10+ years', inplace=True)
train2_data['work_year'].fillna('10+ years', inplace=True)

work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
train1_data['work_year']  = train1_data['work_year'].map(work_year_map)
train2_data['work_year']  = train2_data['work_year'].map(work_year_map)

train1_data['class'] = train1_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
train2_data['class'] = train2_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

train1_data['employer_type'] = train1_data['employer_type'].map(emp_type_dict)
train2_data['employer_type'] = train2_data['employer_type'].map(emp_type_dict)

train1_data['industry'] = train1_data['industry'].map(industry_dict)
train2_data['industry'] = train2_data['industry'].map(industry_dict)


# In[13]:


# 日期类型：issueDate，earliesCreditLine
#train[cat_features]
# 转换为pandas中的日期类型
test_data['issue_date'] = pd.to_datetime(test_data['issue_date'])
# 提取多尺度特征
test_data['issue_date_y'] = test_data['issue_date'].dt.year
test_data['issue_date_m'] = test_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
test_data['issue_date_diff'] = test_data['issue_date'].apply(lambda x: x-base_time).dt.days
test_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
test_data.drop('issue_date', axis = 1, inplace = True)
test_data['work_year'].fillna('10+ years', inplace=True)

work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
test_data['work_year']  = test_data['work_year'].map(work_year_map)
test_data['class'] = test_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
test_data['employer_type'] = test_data['employer_type'].map(emp_type_dict)
test_data['industry'] = test_data['industry'].map(industry_dict)


# ## 模型使用
# 1) LigthGBM
# 2) NN

# ##### 使用internet和bank数据共同特征总量训练

# In[16]:


import lightgbm
from sklearn import metrics

X_train1 = train1_data.drop(['is_default','earlies_credit_mon','loan_id','user_id','region'], axis = 1, inplace = False)
y_train1 = train1_data['is_default']

X_train2 = train2_data.drop(['is_default','earlies_credit_mon','loan_id','user_id','region'], axis = 1, inplace = False)
y_train2 = train2_data['is_default']

X_train = pd.concat([X_train1, X_train2])
y_train = pd.concat([y_train1, y_train2])

X_test = test_data.drop(['earlies_credit_mon','loan_id','user_id','region'], axis = 1, inplace = False)

# 利用Internet数据预训练模型1
clf_ex=lightgbm.LGBMClassifier(n_estimators = 200)
clf_ex.fit(X = X_train, y = y_train)
clf_ex.booster_.save_model('LGBMLGBMClassifier.txt')
pred = clf_ex.predict_proba(X_test)
pred = pred[:,1]


# In[17]:


# submission
submission = pd.DataFrame({'id':test['loan_id'], 'isDefault':pred})
submission.to_csv('submission.csv', index = None)