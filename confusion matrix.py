#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb


# In[3]:


df1=pd.read_csv(r"D:\first sem\Opti poster\train_transaction.csv")


# In[4]:


df1.shape


# In[5]:


df = pd.get_dummies(df1, columns=['ProductCD', 'card4', 'card6', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'])


# In[6]:


df.shape


# In[7]:


df.drop(['P_emaildomain'], axis=1, inplace=True)


# In[8]:


df.isFraud.value_counts(dropna=False)


# In[9]:


train,test=train_test_split(df,test_size=0.3,shuffle=False,random_state=22)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[10]:


train.shape


# In[11]:


test.shape


# In[12]:


train.isFraud.value_counts(dropna=False)


# In[15]:


test.isFraud.value_counts(dropna=False)


# In[13]:


only_zero=train[train.isFraud==0].sample(15000)
only_one=train[train.isFraud==1]
train=pd.concat([only_one,only_zero])


# In[14]:


x_train = train.loc[:, train.columns != 'isFraud']
y_train = train.isFraud

x_test = test.loc[:, test.columns != 'isFraud']
y_test = test.isFraud

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(x_train, y_train) 


# In[15]:


y_pred = xgb_model.predict(x_test)


# In[16]:


print(accuracy_score(y_pred,y_test))


# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[20]:


confusion_matrix_weights=[[0,-10],[-50,20]]


# In[21]:


net_ben=0
for i in range(0,2):
    for j in range(0,2):
        net_ben += confusion_matrix_weights[i][j]*cm[i][j]
net_ben

