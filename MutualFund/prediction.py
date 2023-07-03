import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import eli5
import pickle
from EDA import processed_data

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)

df_train = processed_data()

X = df_train.copy()
X.drop(['Client', 'Sale_MF', 'Revenue_MF'], inplace=True, axis=1)
y_sale_mf = df_train.iloc[:, 28].values
X_train, X_test, y_train, y_test = train_test_split(X, y_sale_mf, test_size=0.2, stratify=y_sale_mf, random_state=1)

# imputing with KNNImputer
col_train = X_train.columns
k = math.sqrt(X_train.shape[0])
imputer = KNNImputer(n_neighbors=round(k), weights='uniform', metric='nan_euclidean')
X_train.Age = X_train.Age.mask(X_train.Age <= 10)
X_train[col_train] = imputer.fit_transform(X_train.values)

X_test.Age = X_test.Age.mask(X_test.Age <= 10)
X_test[col_train] = imputer.transform(X_test.values)

# creating a pickle of KNNImputer to use it for our recomendations
pickle.dump(imputer, open('KNNImputer_MF.pkl', 'wb'))

cv_lr = RepeatedStratifiedKFold(n_splits=20, n_repeats=3, random_state=1)
param_grid_lr = {'m__max_iter': [40, 70, 100],
                 'm__penalty': ['l1', 'l2', 'elasticnet'],
                 'm__class_weight': [{0: 0.20, 1: 0.80}]
                 }
model = LogisticRegression()
Steps_lr = [
    ('msc', MinMaxScaler(feature_range=(0, 1))),
    ('m', model)
]
pipeline_lr = Pipeline(steps=Steps_lr)
grid_search_lr = GridSearchCV(estimator=pipeline_lr, param_grid=param_grid_lr, refit='precision',
                              scoring=['precision', 'recall'], cv=cv_lr, n_jobs=-1, verbose=2)

grid_search_lr.fit(X_train, y_train)
grid_search_lr.best_params_

model_lr = grid_search_lr.best_estimator_

# predicting the values using X_test data set
y_pred_lr = model_lr.predict(X_test)

# feature importance code
importance = model_lr.named_steps['m'].coef_[0]
cols = list(X_train.columns)

# Saving model to disk for sale of consumer loan prediction
pickle.dump(model_lr, open('model_lr_sale_mf.pkl', 'wb'))

X_reg = df_train.copy()
X_reg.drop(['Client', 'Revenue_MF'], inplace=True, axis=1)
y_revenue_mf = df_train.iloc[:, 29].values
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_revenue_mf, test_size=0.2, random_state=1)

param_grid_rf = {
    'm__max_depth': [80, 90],
    'm__min_samples_leaf': [4, 5],
    'm__min_samples_split': [8, 10],
    'm__n_estimators': [50, 100, 150]
}
model = RandomForestRegressor()
Steps_rf = [('m', model)]
pipeline_rf = Pipeline(steps=Steps_rf)
grid_search_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, refit='neg_mean_absolute_error',
                              scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'], cv=10, n_jobs=-1,
                              verbose=2)

grid_search_rf.fit(X_train_reg, y_train_reg)
grid_search_rf.best_params_

model_rf = grid_search_rf.best_estimator_

# predicting the values using X_test data set
y_pred_rf = model_rf.predict(X_test_reg)

# Calculating the metrics for our model performance
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_reg, y_pred_rf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_reg, y_pred_rf))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_reg, y_pred_rf)))

# Feature importance
cols_reg = list(X_train_reg.columns)
eli5.explain_weights(model_rf.named_steps['m'], top=15, feature_names=cols_reg)

# Saving model to disk for revenue prediction
pickle.dump(model_rf, open('model_rf_revenue_mf.pkl', 'wb'))

df_test = pd.read_excel("Data/testDatasetCreation/test.xlsx",engine='openpyxl')

df_test_client = df_test["Client"]

df_test.drop(['Unnamed: 0', 'Client'], inplace=True, axis=1)

# dropping columns
columns_sale_mf = ['Count_MF', 'ActBal_MF']
df_test.drop(columns_sale_mf, inplace=True, axis=1)

# # replacing null values with U (Unknown) if any
df_test.Sex = df_test.Sex.replace(np.nan, "U", regex=True)

# converting M and F to 1 and 0
df_test.Sex = df_test.Sex.replace({'M': 1, 'F': 0, 'U': 2})

# imputing age with tenure + 120 months if tenure is more than Age
df_test.Age = np.where((df_test.Age * 12 <= df_test.Tenure), round(df_test.Tenure / 12) + 10, df_test.Age)

# imputing other null values with 0 in the data set
df_test.fillna(0, inplace=True)

col_test = df_test.columns
df_test.Age = df_test.Age.mask(df_test.Age <= 10)
df_test[col_test] = imputer.transform(df_test.values)

y_pred_lr_mf = model_lr.predict(df_test)

# predicting the values using X_test data set
y_pred_lr_mf_prob = model_lr.predict_proba(df_test) #logistic regression is used

df_test['Sale_MF'] = y_pred_lr_mf

y_pred_rf_mf = model_rf.predict(df_test) #random forest regressor

df_pred = pd.DataFrame({'Client': df_test_client, 'ProbablitySaleMF': y_pred_lr_mf_prob[:, 1].reshape(-1),
                        'RevenueMF': y_pred_rf_mf.reshape(-1)})

df_pred[df_pred['ProbablitySaleMF'] > 0.5]

df_pred.to_excel("Data/df_pred.xlsx")

