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
from sklearn.ensemble import RandomForestClassifier

#SearchGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

#Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#Split
from sklearn.model_selection import train_test_split

#Resample
from sklearn.utils import resample

#Saving 
from sklearn.externals import joblib 

#################################################################
#Prep train/validation set from external file
#################################################################
exec(open('../scripts/dataprep_treino_validacao.py').read())
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
##################################################################
#Param grid
Cs = np.arange(0.5,100,0.5)
gammas = [0.001, 0.01, 0.1, 1]

param_grid = {'C':Cs,'gamma':gammas}
#################################################################
#Grid Search
print('Tuning RBF Kernel parameters')
grid_search = GridSearchCV(svm.SVC(kernel='rbf'),param_grid)
grid_search.fit(X_train,y_train)
print('Search grid for RBF returned parameters:')
print(grid_search.best_params_)
#Get model  
SVCrbf = grid_search.best_estimator_
print('Params:')
print(SVCrbf.get_params())
#SVCrbf.fit(X_train,y_train)
print('Accuracy score of RBF Kernel:',SVCrbf.score(X_valid,y_valid))
joblib.dump(SVCrbf, '../models/svcrbf.pkl') 
##################################################################
#Re-treino com validação
##################################################################
print('Re-training with validation data')
X = np.concatenate((X_train,X_valid))
y = np.concatenate((y_train,y_valid))

SVCrbf = joblib.load('../models/svcrbf.pkl')
SVCrbf.fit(X,y)
#Scoring
scores = cross_val_score(SVCrbf, X, y, cv = 3)
print('Training + Validation score(cross-val)',scores.mean())

##################################################################
##################################################################
##################################################################
#Paradigma One vs All
##################################
print('Implementing OneVsAll')
#Separando uma coluna para cada target
for target in df['label'].unique():
    df[target] = (df['label']==target).astype(int)
    df_train[target] = (df_train['label']==target).astype(int)
    df_valid[target] = (df_valid['label']==target).astype(int)
#Treinando um modelo para cada classe
models={}
    
models['geral'] = joblib.load('../models/SVCrbf.pkl')

backup_X_train = X_train.copy()
backup_y_train = y_train.copy()
backup_X_valid = X_valid.copy()
backup_y_valid = y_valid.copy()

for target in df['label'].unique():
    
    X_train = backup_X_train.copy()
    y_train = backup_y_train.copy()
    X_valid = backup_X_valid.copy()
    y_valid = backup_y_valid.copy()
    
    train_temp = pd.DataFrame(X_train)
    train_temp[str(target)]=df_train[target].copy()
    
    train_temp_nao_target = train_temp.loc[train_temp[target]==0]
    train_temp_target = train_temp.loc[train_temp[target]==1]
    
    train_temp_upsampled = resample(train_temp_target,
                          replace=True, # sample with replacement
                          n_samples=int(len(train_temp_nao_target)/2), # match number in majority class
                          random_state=27) # reproducible results
    
    # combine majority and upsampled minority
    train_temp = pd.concat([train_temp_nao_target, train_temp_upsampled])
    
    X_train = train_temp.iloc[:,:-1]
    y_train = train_temp.iloc[:,-1]
    
    #del SVCrbf
    SVCrbf = svm.SVC(
        kernel='rbf',
        gamma=0.1,
        C=3
    )
    #print(X_train.shape)
    #print(y_train.shape)
    
    SVCrbf.fit(X_train,y_train)
    print("**********************************************************************")
    print("Acuracia para a classe "+target+":",SVCrbf.score(X_valid,df_valid[target]))
    predictions = SVCrbf.predict(X_valid)
    print("Roc AUC score para a classe "+target+":",roc_auc_score(df_valid[target],predictions))
    print("Precision:",precision_score(df_valid[target],predictions))
    print("Recall:",recall_score(df_valid[target],predictions))
    
    #For re-training, oversample will also be used
    train_temp = pd.DataFrame(X_valid)
    train_temp[str(target)]=df_valid[target].copy()
    
    train_temp_nao_target = train_temp.loc[train_temp[target]==0]
    train_temp_target = train_temp.loc[train_temp[target]==1]
    
    train_temp_upsampled = resample(train_temp_target,
                          replace=True, # sample with replacement
                          n_samples=int(len(train_temp_nao_target)/2), # match number in majority class
                          random_state=27) # reproducible results
    
    # combine majority and upsampled minority
    train_temp = pd.concat([train_temp_nao_target, train_temp_upsampled])
    
    X_valid2 = train_temp.iloc[:,:-1]
    y_valid2 = train_temp.iloc[:,-1]
    
    X = np.concatenate((X_train,X_valid2))
    y = np.concatenate((y_train,y_valid2))
    SVCrbf.fit(X,y)
    #Scoring
    scores = cross_val_score(SVCrbf, X, y, cv = 3)
    print('Training + Validation score(cross-val) para classe '+target+':',scores.mean())
    joblib.dump(SVCrbf,'../models/SVCrbf_'+target+'.pkl')
    models[target]=joblib.load('../models/SVCrbf_'+target+'.pkl')
    
print("**********************************************************************")