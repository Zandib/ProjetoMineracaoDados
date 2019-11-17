# -*- coding: utf-8 -*-
"""
@author: dib_n
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import librosa
import noisereduce as nr
import joblib

import warnings
warnings.filterwarnings('ignore')

df = pd.DataFrame([[1,2,3,4,5,6,7,8,9,10]],columns = 'a,b,c,d,h,m,n,x,6,7'.split(','))

for letter in 'a,b,c,d,h,m,n,x,6,7'.split(','):
    t_path = '../treinamento/'+letter
    onlyfiles = [f for f in listdir(t_path) if isfile(join(t_path, f))]
    df[letter]=len(onlyfiles)

df.index=['Número de arquivos']

print('Dados de treino')
print(df)

t_path = '../treinamento'
onlyfiles = [f for f in listdir(t_path) if isfile(join(t_path, f))]
df = pd.DataFrame([[1,2,3,4,5,6,7,8,9,10]],columns = 'a,b,c,d,h,m,n,x,6,7'.split(','))

for letter in 'a,b,c,d,h,m,n,x,6,7'.split(','):
    t_path = '../validacao/'+letter
    onlyfiles = [f for f in listdir(t_path) if isfile(join(t_path, f))]
    df[letter]=len(onlyfiles)

df.index=['nmr de arquivos']

print('Dados de validacao')
print(df)

#Imagem de exemplo
data_test, fs = librosa.load('../treinamento/a/1_3.wav')
plt.figure(figsize=(15,5))
plt.plot(data_test)
plt.title('Original data')
plt.savefig('../imagens/original_data.png')
plt.show()
#Reducao de ruido
reduced_noise = nr.reduce_noise(audio_clip=data_test, noise_clip=data_test[1750:], verbose=False)
plt.figure(figsize=(15,5))
plt.plot(reduced_noise)
plt.title('Noise Reducted')
plt.savefig('../imagens/noise_reducted.png')
plt.show()
#Normalizacao
plt.figure(figsize=(15,5))
plt.plot(librosa.util.normalize(reduced_noise))
plt.title('Normalized')
plt.ylim(-1,1)
plt.savefig('../imagens/normalized.png')
plt.show()

##Exemplo de tabela de dados
dados = pd.read_csv('../data/data.csv',index_col=False)
print(dados.head())

#Carregar modelos
SVM = joblib.load('../models/svcrbf_2.pkl')
SVMlinear = joblib.load('../models/svclinear.pkl')
scaler = joblib.load('../models/scaler_2.pkl')
encoder = joblib.load('../models/encoder_2.pkl')

#Carregar dados de validacao
dado_valid = pd.read_csv('../data/data_validacao.csv',index_col=False)
labels = dado_valid.copy()['label']
dado_valid.head()
#Excluindo colunas desnecessarias
dado_valid = dado_valid.drop('filename',axis=1)
dado_valid = dado_valid.drop('label',axis=1)
#Scailing
dado_valid = scaler.transform(dado_valid)
#Prediction
predictions = SVM.predict(dado_valid)
predictions = encoder.inverse_transform(predictions)
predictions_linear = SVMlinear.predict(dado_valid)
predictions_linear = encoder.inverse_transform(predictions_linear)
#Criando o df
df = pd.DataFrame(columns=['Real','Predicao_rbf'])
df['Real']=labels
df['Predicao_rbf'] = predictions
df['Predicao_linear'] = predictions_linear
df['Acerto_linear'] = (df['Real']==df['Predicao_linear']).astype(int)
df['Erro_linear'] =  (df['Real']!=df['Predicao_linear']).astype(int)
df['Acerto_rbf'] = (df['Real']==df['Predicao_rbf']).astype(int)
df['Erro_rbf'] =  (df['Real']!=df['Predicao_rbf']).astype(int)
compilado = df[['Real','Acerto_rbf','Erro_rbf','Acerto_linear','Erro_linear']].groupby('Real').sum().transpose()
compilado['Total'] = compilado.sum(axis=1)
compilado['Total(%)'] = compilado['Total']/1068
pd.set_option('max_columns',30)
print(compilado)
#Criando gráfico
plt.figure(figsize=(15,7))
plt.title('Número de Acertos por Caracter')
plt.bar(encoder.classes_,compilado.transpose()['Acerto_rbf'][:-2],label='RBF')
plt.bar(encoder.classes_,compilado.transpose()['Acerto_linear'][:-2],label='Linear')
plt.legend()
plt.savefig('../imagens/nmr_acerto_p_char')
plt.show()