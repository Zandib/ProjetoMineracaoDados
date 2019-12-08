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
models ={}
for target in ['geral', 'a', 'b', 'c', 'd', 'h', 'm', 'n', 'x', '6', '7']:
    if(target=='geral'):
        models[target]=joblib.load('../models/SVCrbf.pkl')
    else:
        models[target]=joblib.load('../models/SVCrbf_'+target+'.pkl')

#################################################################
#To Stack multiple models, an heuristic will be used
#Heuristica:
class Voter():
    def __init__(self,models,targets,encoder):
        self.targets=targets
        self.models=models
        self.encoder = encoder
    
    def check_line(self,x):
        x=x.copy()
        #print(x)
        x_temp = x.drop('geral')
        if(x_temp.sum()==1):
            for i,t in zip(x_temp.index,x_temp):
                #print(i)
                if(t==1):
                    return i
        else:
            return x['geral']
        
    def Score(self,X,y):
        predicts = pd.Series(self.predict(X))
        return (predicts==y).value_counts()[True]/len(y)
    
    def predict(self,X):
        df_predicoes = pd.DataFrame(columns=self.targets)
        for t in targets:
            df_predicoes[t]=self.models[t].predict(X)
        df_predicoes['geral'] = encoder.inverse_transform(df_predicoes['geral'])
        df_predicoes['predicts'] = df_predicoes.apply(lambda x: self.check_line(x),axis=1)
        return np.array(df_predicoes['predicts'])
#################################################################

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
targets = ['geral', 'a', 'b', 'c', 'd', 'h', 'm', 'n', 'x', '6', '7']
vt = Voter(models,targets,encoder)
predictions = vt.predict(df_teste)

#predictions = svmrbf.predict(df_teste)
#predictions = encoder.inverse_transform(predictions)
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