# -*- coding: utf-8 -*-
"""
@author: dib_n
"""
#################################################################
#Imports
#################################################################
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Learning
from sklearn import svm

#SearchGrid
from sklearn.model_selection import GridSearchCV

#Saving 
from sklearn.externals import joblib 

#################################################################
#Prep train/validation set from external file
#################################################################
exec(open('dataprep_treino_validacao.py').read())
#################################################################
#Loads
#################################################################
#Importando base de treino
df_train = pd.read_csv('../data/data.csv',index_col=False)
df_valid = pd.read_csv('../data/data_validacao.csv',index_col=False)
# Dropping unneccesary columns
df_train = df_train.drop(['filename'],axis=1)
df_valid = df_valid.drop('filename',axis=1)
#################################################################
#Encoding
#################################################################
class_list = df_train.iloc[:, -1]
encoder = LabelEncoder()
encoder.fit(class_list)
y_train = encoder.transform(class_list)
y_valid = encoder.transform(df_valid.iloc[:,-1])
joblib.dump(encoder,'../models/encoder.pkl')
#################################################################
#Scailing
#################################################################
scaler = StandardScaler()
df = df_train.append(df_valid,ignore_index=True)
scaler.fit(np.array(df.iloc[:, :-1], dtype = float))
X_train = scaler.transform(np.array(df_train.iloc[:, :-1], dtype = float))
X_valid = scaler.transform(np.array(df_valid.iloc[:, :-1], dtype = float))
joblib.dump(scaler,'../models/scaler.pkl')
#################################################################
#Training svm with two different kernels
#################################################################
#Linear
SVClassifier = svm.SVC(kernel='linear')
SVClassifier.fit(X_train,y_train)
print('Accuracy of linear SVM:',SVClassifier.score(X_valid,y_valid))
joblib.dump(SVClassifier,'../models/svclinear.pkl')
#Rbf
##Param grid
Cs = np.arange(0.5,100,0.5)
gammas = [0.001, 0.01, 0.1, 1]

param_grid = {'C':Cs,'gamma':gammas}
#Grid Search
print('Tuning RBF Kernel parameters')
grid_search = GridSearchCV(svm.SVC(kernel='rbf'),param_grid)
grid_search.fit(X_train,y_train)
print('Search grid for RBF returned parameters:')
print(grid_search.best_params_)
#Get model  
SVCrbf = grid_search.best_estimator_
print('Accuracy score of RBF Kernel:',SVCrbf.score(X_valid,y_valid))
joblib.dump(SVCrbf, '../models/svcrbf.pkl') 