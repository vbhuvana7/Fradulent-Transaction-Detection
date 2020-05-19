import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb

df1=pd.read_csv(r"D:\first sem\Opti poster\train_transaction.csv")
df1.shape
#one hot encoding for all the categorical variables
df = pd.get_dummies(df1, columns=['ProductCD', 'card4', 'card6', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'])
#dropping P_emaildomain as it has many different values
df.drop(['P_emaildomain'], axis=1, inplace=True)

df.isFraud.value_counts(dropna=False)

train,test=train_test_split(df,test_size=0.3,shuffle=False,random_state=22)

train.isFraud.value_counts(dropna=False)

test.isFraud.value_counts(dropna=False)

only_zero=train[train.isFraud==0].sample(15000)
only_one=train[train.isFraud==1]
train=pd.concat([only_one,only_zero])

x_train = train.loc[:, train.columns != 'isFraud']
y_train = train.isFraud

x_test = test.loc[:, test.columns != 'isFraud']
y_test = test.isFraud

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(x_train, y_train) 
y_pred = xgb_model.predict(x_test)

print(accuracy_score(y_pred,y_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

confusion_matrix_weights=[[0,-10],[-50,20]]

net_ben=0
for i in range(0,2):
    for j in range(0,2):
        net_ben += confusion_matrix_weights[i][j]*cm[i][j]
net_ben
