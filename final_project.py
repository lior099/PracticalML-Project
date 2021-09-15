import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
# to get top 10 features in dictionary
from heapq import nlargest
from operator import itemgetter

from matplotlib import pyplot


# read data
# df = pd.read_csv("../input/web-page-phishing-detection-dataset/dataset_phishing.csv")
df = pd.read_csv('dataset_phishing.csv')


# delete columns that contain only zero
def zero_column(data):
    # if the sum of the column is different than zero and all the elements are greater or equal to zero
    # then there are other values in the column
    # key: name of column, value: sum of the column
    columns = {}
    # the columns to delete
    column = []

    # run through all the columns
    for col in data.iloc[:,1:].columns:

        # abs on every element
        ab = data[col].abs()
        # the sum of the column
        sm = ab.sum()
        columns[col] = sm

    for key in columns.keys():

        # the sum in this column is 0. need to delete
        if columns[key] == 0:
            column.append(key)

    # exclude the columns from the data
    new_df = data.drop(column, axis=1)

    return new_df


def binary_column(data):

    # changing 'status' column to binary
    data["status"] = data["status"].map({"legitimate": 0, "phishing": 1})

    return data


def split_y(data):

    # x has the features
    x = data.iloc[:, 1:-1]
    # binary
    y = data["status"]

    return x, y


# norm = mm means normalize using min max
# norm = std means normalize using standard
def normalize(x_train_data, x_test_data, norm):

    if norm == 'mm':

        scaler = MinMaxScaler().fit(x_train_data)
        x_train = scaler.transform(x_train_data)
        x_test = scaler.transform(x_test_data)

    else:
        scaler = StandardScaler().fit(x_train_data)
        x_train = scaler.transform(x_train_data)
        x_test = scaler.transform(x_test_data)

    return x_train, x_test


def correlation_figure(data):
    # correlation
    corr = data.corr()
    fig = plt.figure(figsize=(20, 20))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='BuPu', robust=True, center=0, square=True, linewidths=.5, vmin=-1.0, vmax=1.0,
                cbar_kws={"shrink": 0.5})
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    fig.savefig('Correlations new')


############################################################################
"""
df = binary_column(df)  # changing status column to binary

df_new = zero_column(df)  # exclude columns that contain only zeros

# correlation on the whole data with y vector
correlation_figure(df_new)

x, y = split_y(df_new)  # separate the status column from the rest of the data

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)


# normalized with min max
train_x_mm, test_x_mm = normalize(train_x, test_x, 'mm')

# normalized with standard
train_x_std, test_x_std = normalize(train_x, test_x, 'std')
"""


df = binary_column(df)  # changing status column to binary

df_new = zero_column(df)  # exclude columns that contain only zeros

x, y = split_y(df_new)  # separate the status column from the rest of the data


# correlation_figure(df_new)

# figure of one row of correlation with threshold
"""
column = []
corr = df_new.corr()
status_corr = corr.tail(1)
status_corr = status_corr.drop(['status'], axis=1)

for col in status_corr.iloc[:, 1:].columns:
    if -0.2 < status_corr[col].item() < 0.2:
        column.append(col)
        
new_status_corr = status_corr.drop(column, axis=1)
sort_corr = new_status_corr.sort_values(by='status', axis=1, ascending=False)

fig = plt.figure(figsize=(20, 20))
sns.set(font_scale=1.9)
sns.heatmap(sort_corr, cmap='BuPu', robust=True, center=0, square=True, linewidths=.5, vmin=-1.0, vmax=1.0,
            cbar_kws={"shrink": 0.3, "ticks": [-1, 0, 1]})
plt.yticks(fontsize=30, rotation=360)
plt.xticks(fontsize=30)
fig.savefig('correlation - 24 features')
"""

# logistic regression ###############################

# define the model
model_lr = LogisticRegression(C=8.659906107728098, max_iter=1000, solver='liblinear')

feature_importance_lr = {}
# get name of features
feature_list_lr = x.columns.tolist()
top_10_features_lr = {}

# fit the model
model_lr.fit(x, y)

# get importance
importance_lr = model_lr.coef_[0]
importance_lr = abs(importance_lr)

# summarize feature importance
for i, v in zip(feature_list_lr, importance_lr):
    print('Feature: , Score: ', (i, v))
    feature_importance_lr[i] = v
    
# find top 10 features
for feature, score in nlargest(10, feature_importance_lr.items(), key=itemgetter(1)):
    print(feature, score)
    top_10_features_lr[feature] = score

"""
# save feature importance to excel
df_dict = pd.DataFrame(data=feature_importance_lr, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('logistic regression feature importance.xlsx')

# plot feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar([x for x in range(len(importance_lr))], importance_lr)
plt.title('logistic_regression importance', fontsize=20, font="Serif")
fig.savefig('logistic_regression importance')

# save top 10 feature importance to excel
df_dict = pd.DataFrame(data=top_10_features_lr, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('top 10 logistic regression feature importance.xlsx')

# plot top 10 feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar(*zip(*top_10_features_lr.items()))
plt.title('Top 10 logistic_regression importance', fontsize=20, font="Serif")
fig.savefig('top 10 logistic_regression importance')
"""
"""
# plot top 10 feature importance
fig = plt.figure(figsize=(25, 25))
pyplot.bar(*zip(*top_10_features_lr.items()), color='red')
#plt.tight_layout()
plt.xticks(fontsize=50, rotation=90)
# plt.tight_layout()
plt.yticks(fontsize=50)
plt.title('Logistic Regression importance', fontsize=100, font="Serif")
plt.tight_layout()
fig.savefig('logistic_regression importance new')
"""
# random forest########################################

# define the model
model_rf = RandomForestClassifier(max_depth=50, min_samples_leaf=2)

feature_importance_rf = {}
# get name of features
feature_list_rf = x.columns.tolist()
top_10_features_rf = {}

# fit the model
model_rf.fit(x, y)
# get importance
importance_rf = model_rf.feature_importances_

# summarize feature importance
for i, v in zip(feature_list_rf, importance_rf):
    print('Feature: , Score: ', (i, v))
    feature_importance_rf[i] = v

# find top 10 features
for feature, score in nlargest(10, feature_importance_rf.items(), key=itemgetter(1)):
    print(feature, score)
    top_10_features_rf[feature] = score

"""
# save feature importance to excel
df_dict = pd.DataFrame(data=feature_importance_rf, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('RandomForestClassifier feature importance.xlsx')

# plot feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar([x for x in range(len(importance_rf))], importance_rf)
plt.title('RandomForestClassifier importance', fontsize=20, font="Serif")
fig.savefig('RandomForestClassifier importance')
"""
"""
# save top 10 feature importance to excel
df_dict = pd.DataFrame(data=top_10_features_rf, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('top 10 RandomForestClassifier feature importance.xlsx')

# plot top 10 feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar(*zip(*top_10_features_rf.items()))
plt.title('Top 10 RandomForestClassifier importance', fontsize=20, font="Serif")
fig.savefig('top 10 RandomForestClassifier importance')
"""
"""
# plot top 10 feature importance
fig = plt.figure(figsize=(25, 25))
pyplot.bar(*zip(*top_10_features_rf.items()), color='red')
#plt.tight_layout()
plt.xticks(fontsize=50, rotation=90)
# plt.tight_layout()
plt.yticks(fontsize=50)
plt.title('Random Forest importance', fontsize=100, font="Serif")
plt.tight_layout()
fig.savefig('Random Forest importance new')
"""
# GradientBoostingClassifier ######################################################

# define the model
model_gbc = GradientBoostingClassifier(learning_rate=0.04615056515636364, max_depth=10, n_estimators=1000, verbose=1)

feature_importance_gbc = {}
# get name of features
feature_list_gbc = x.columns.tolist()
top_10_features_gbc = {}

# fit the model
model_gbc.fit(x, y)

# get importance
importance_gbc = model_gbc.feature_importances_

# summarize feature importance
for i, v in zip(feature_list_gbc, importance_gbc):
    print('Feature: , Score: ', (i, v))
    feature_importance_gbc[i] = v

# find top 10 features
for feature, score in nlargest(10, feature_importance_gbc.items(), key=itemgetter(1)):
    print(feature, score)
    top_10_features_gbc[feature] = score

"""
# save feature importance to excel
df_dict = pd.DataFrame(data=feature_importance_gbc, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('GradientBoostingClassifierr feature importance.xlsx')

# plot feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar([x for x in range(len(importance_gbc))], importance_gbc)
plt.title('GradientBoostingClassifier importance', fontsize=20, font="Serif")
fig.savefig('GradientBoostingClassifier importance')


# save top 10 feature importance to excel
df_dict = pd.DataFrame(data=top_10_features_gbc, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('top 10 GradientBoostingClassifier feature importance.xlsx')

# plot top 10 feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar(*zip(*top_10_features_gbc.items()))
plt.title('Top 10 GradientBoostingClassifier importance', fontsize=20, font="Serif")
fig.savefig('top 10 GradientBoostingClassifier importance')
"""
"""
# plot top 10 feature importance
fig = plt.figure(figsize=(25, 25))
pyplot.bar(*zip(*top_10_features_gbc.items()), color='red')
#plt.tight_layout()
plt.xticks(fontsize=50, rotation=90)
# plt.tight_layout()
plt.yticks(fontsize=50)
plt.title('Gradient Boosting importance', fontsize=100, font="Serif")
plt.tight_layout()
fig.savefig('Gradient Boosting importance new')
"""
# DecisionTreeClassifier ###################################################

# define the model
model_dtc = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)

feature_importance_dtc = {}
# get name of features
feature_list_dtc = x.columns.tolist()
top_10_features_dtc = {}

# fit the model
model_dtc.fit(x, y)

# get importance
importance_dtc = model_dtc.feature_importances_

# summarize feature importance
for i, v in zip(feature_list_dtc, importance_dtc):
    print('Feature: , Score: ', (i, v))
    feature_importance_dtc[i] = v

# find top 10 features
for feature, score in nlargest(10, feature_importance_dtc.items(), key=itemgetter(1)):
    print(feature, score)
    top_10_features_dtc[feature] = score

"""
# save feature importance to excel
df_dict = pd.DataFrame(data=feature_importance_dtc, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('DecisionTreeClassifier feature importance.xlsx')

# plot feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar([x for x in range(len(importance_dtc))], importance_dtc)
plt.title('DecisionTreeClassifier importance', fontsize=20, font="Serif")
fig.savefig('DecisionTreeClassifier importance')
"""
"""
# save top 10 feature importance to excel
df_dict = pd.DataFrame(data=top_10_features_dtc, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('top 10 DecisionTreeClassifier feature importance.xlsx')

# plot top 10 feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar(*zip(*top_10_features_dtc.items()))
plt.title('Top 10 DecisionTreeClassifier importance', fontsize=20, font="Serif")
fig.savefig('top 10 DecisionTreeClassifier importance')
"""
"""
# plot top 10 feature importance
fig = plt.figure(figsize=(25, 25))
pyplot.bar(*zip(*top_10_features_dtc.items()), color='red')
#plt.tight_layout()
plt.xticks(fontsize=50, rotation=90)
# plt.tight_layout()
plt.yticks(fontsize=50)
plt.title('Decision Tree importance', fontsize=100, font="Serif")
plt.tight_layout()
fig.savefig('Decision Tree importance new')
"""

# XGBClassifier #################################################################

# define the model
model_xgb = xgb.XGBClassifier(use_label_encoder=False, learning_rate=0.10872699850937145, max_depth=20, n_estimators=500
                              , verbosity=1, subsample=0.7)

feature_importance_xgb = {}
# get name of features
feature_list_xgb = x.columns.tolist()
top_10_features_xgb = {}

# fit the model
model_xgb.fit(x, y)

# get importance
importance_xgb = model_xgb.feature_importances_

# summarize feature importance
for i, v in zip(feature_list_xgb, importance_xgb):
    print('Feature: , Score: ', (i, v))
    feature_importance_xgb[i] = v

# find top 10 features
for feature, score in nlargest(10, feature_importance_xgb.items(), key=itemgetter(1)):
    print(feature, score)
    top_10_features_xgb[feature] = score

"""
# save feature importance to excel
df_dict = pd.DataFrame(data=feature_importance_xgb, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('XGBClassifier feature importance.xlsx')

# plot feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar([x for x in range(len(importance_xgb))], importance_xgb)
plt.title('XGBClassifier importance', fontsize=20, font="Serif")
fig.savefig('XGBClassifier importance')

# save top 10 feature importance to excel
df_dict = pd.DataFrame(data=top_10_features_xgb, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('top 10 XGBClassifier feature importance.xlsx')

# plot top 10 feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar(*zip(*top_10_features_xgb.items()))
plt.title('Top 10 XGBClassifier importance', fontsize=20, font="Serif")
fig.savefig('top 10 XGBClassifier importance')
"""
"""
# plot top 10 feature importance
fig = plt.figure(figsize=(15, 15))
pyplot.bar(*zip(*top_10_features_xgb.items()), color='red')
# plt.tight_layout()
plt.xticks(fontsize=25, rotation=90)
# plt.tight_layout()
plt.yticks(fontsize=20)
plt.title('XGB importance', fontsize=100, font="Serif")
plt.tight_layout()
fig.savefig('XGB importance1')
"""

# AdaBoostClassifier #######################################################

# define the model
model_abc = AdaBoostClassifier(learning_rate=0.48416301546239815, n_estimators=1000)

feature_importance_abc = {}
# get name of features
feature_list_abc = x.columns.tolist()
top_10_features_abc = {}

# fit the model
model_abc.fit(x, y)

# get importance
importance_abc = model_abc.feature_importances_

# summarize feature importance
for i, v in zip(feature_list_abc, importance_abc):
    print('Feature: , Score: ', (i, v))
    feature_importance_abc[i] = v

# find top 10 features
for feature, score in nlargest(10, feature_importance_abc.items(), key=itemgetter(1)):
    print(feature, score)
    top_10_features_abc[feature] = score

"""
# save feature importance to excel
df_dict = pd.DataFrame(data=feature_importance_abc, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('AdaBoostClassifier feature importance.xlsx')

# plot feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar([x for x in range(len(importance_abc))], importance_abc)
plt.title('AdaBoostClassifier importance', fontsize=20, font="Serif")
fig.savefig('AdaBoostClassifier importance')

# save top 10 feature importance to excel
df_dict = pd.DataFrame(data=top_10_features_abc, index=[0])
# transpose
df_dict = df_dict.T
df_dict.to_excel('top 10 AdaBoostClassifier feature importance.xlsx')

# plot top 10 feature importance
fig = plt.figure(figsize=(20, 20))
pyplot.bar(*zip(*top_10_features_abc.items()))
plt.title('Top 10 AdaBoostClassifier importance', fontsize=20, font="Serif")
fig.savefig('top 10 AdaBoostClassifier importance')
"""
"""
# plot top 10 feature importance
fig = plt.figure(figsize=(25, 25))
pyplot.bar(*zip(*top_10_features_abc.items()), color='red')
# plt.tight_layout()
plt.xticks(fontsize=50, rotation=90)
# plt.tight_layout()
plt.yticks(fontsize=50)
plt.title('Ada Boost importance', fontsize=100, font="Serif")
plt.tight_layout()
fig.savefig('AdaBoost importance new')
"""
