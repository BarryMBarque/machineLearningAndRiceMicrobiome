#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


# In[ ]:


np.random.seed(42)


# In[2]:


def Perceptron_Model(X_train, y_train, X_test):
    clp= Perceptron().fit(X_train, y_train)
    mlp_pred = clp.predict(X_test)
    return mlp_pred, clp


# In[3]:


def Decision_Tree_Model(X_train, y_train, X_test):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_predict_dt = clf.predict(X_test)
    return y_predict_dt, clf


# In[4]:


def Random_Forest_Model(X_train, y_train, X_test):
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)
    y_predict_rf = clr.predict(X_test)
    return y_predict_rf, clr


# In[5]:


def SVM_Model(X_train, y_train, X_test):
    cls = make_pipeline(StandardScaler(), SVC())
    cls.fit(X_train, y_train)
    y_predict_svm = cls.predict(X_test)
    return y_predict_svm, cls

