# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 07:39:09 2024

@author: shubh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###########################       Step 1       ################################
########################### Importing Data Set ################################

# Importing the dataset 
dataset = pd.read_csv('credit.csv')
#x = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, -2].values

# Printing the data set
print(dataset.head())

# Printing all the column names 
column_headers = list(dataset.columns.values)
print("The Column Header :", column_headers)

###########################       Step 2       ################################
###########################  Cleaning Data Set ################################


# Checking if data set has null values 
print(dataset.isnull().sum())

# Collecting all the null values present
all_rows = dataset[dataset.isnull().any(axis=1)] 

#deleting duplicate values
# Duplicate values 
dupilcate = dataset[dataset.duplicated()]

#deleting duplicate values
dataset.drop_duplicates(inplace=True)

# Droping unwanted coulmns 
dataset = dataset.drop(['step','nameOrig','nameDest'],axis=1) #### drops a column

# If we need to count the total number of transactions
print(dataset.type.value_counts())


###########################       Step 3       ################################
###########################  Feature Selction  ################################

# Feature selection 
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -2].values


# Using information gain method
from sklearn.feature_selection import mutual_info_classif
importance = mutual_info_classif(x,y)
feat_importance = pd.Series(importance, dataset.columns[1: len(dataset.columns)-1])
km = feat_importance.plot(kind='barh', color='teal',title="Information Gain")
km.figure.savefig('Infomation_gain.png')

# final number of features selected
x_new = dataset.loc[:, ['type','amount','oldbalanceOrg','newbalanceOrig' ]].values
y_new = dataset.loc[:,['isFraud']].values

###########################       Step 4            ###########################
###########################  Categorical variables  ###########################


# Converting categorical variables
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x_new  = np.array(ct.fit_transform(x_new))

###########################       Step 5            ###########################
###########################    Data splitting       ###########################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


###########################       Step 6            ###########################
###########################  Model Implimentation   ###########################

# XGBoost classfifer
from xgboost import XGBClassifier 
classifierXGB = XGBClassifier()
classifierXGB.fit(x_train,y_train)
y_predictXGB = classifierXGB.predict(x_test)

# printing confusion matrix 
from sklearn.metrics import confusion_matrix
cmXGB = confusion_matrix(y_test, y_predictXGB)
print('Confusion Matrix by XGBoost:',cmXGB)

# Accuracy score
from sklearn.metrics import accuracy_score
print('Average acuracy by XGBoost:',accuracy_score(y_test, y_predictXGB))


# Using random forest 
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators = 101, criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)
y_predict = classifier.predict(x_test)

#########confusion matrix##########

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print('Confusion Matrix by Random forest:',cm)
############ accuracy###################
from sklearn.metrics import accuracy_score
print('Accuracy by Random forest:',accuracy_score(y_test, y_predict))

###########################       Step 6            ###########################
###########################    Model Selection      ###########################

# K- validation
from sklearn.model_selection import cross_val_score
accuracy_k = cross_val_score(estimator= classifier,X = x_train, y = y_train, cv = 10, n_jobs = -1) # use n_job =-1 to use cpus
print(' Average acuracy by K fold cross validation: ', accuracy_k.mean())
print('Std acuracy by K fold cross validation:',accuracy_k.std())
