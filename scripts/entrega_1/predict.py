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
#Loading Models
#################################################################
encoder = joblib.load('../models/encoder.pkl')
scaler = joblib.load('../models/scaler.pkl')
svmrbf = joblib.load('../models/svcrbf.pkl')


#################################################################
#Preping data with external file   
#################################################################
exec(open('dataprep_teste.py').read())

#################################################################
#Predicting all files
#################################################################
df_teste = pd.read_csv('../data/data_teste.csv')
df_raw = df_teste.copy()
#Dropping filename for predicitons
df_teste = df_teste.drop('filename',axis=1)
#Scailing
df_teste = scaler.transform(df_teste)
#Predicting
predictions = svmrbf.predict(df_teste)
predictions = encoder.inverse_transform(predictions)
#print(df_raw['filename'])

#################################################################
#Assembling
#################################################################
df_respostas = pd.DataFrame(columns=['filename_extended'])
df_respostas['filename_extended'] = df_raw['filename']
df_respostas['predictions'] = predictions


df_respostas['filename'] = df_respostas['filename_extended'].apply(lambda x: x.split('_')[0])
df_respostas['file_pos'] = df_respostas['filename_extended'].apply(lambda x: x.split('_')[2])
df_respostas['file_pos'] = df_respostas['file_pos'].str.replace('.wav','')

df_final = pd.DataFrame(
    df_respostas[['filename','predictions']].groupby('filename').predictions.apply(lambda x: x.sum())
).reset_index()

print(df_final)
df_final.to_csv('../data/resultado_predicoes.csv',index=False)
print('Resultados salvos no arquivo "resultado_predicoes.csv"')