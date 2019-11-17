# -*- coding: utf-8 -*-
"""
@author: dib_n
"""

###################################
#Imports
###################################
# feature extractoring and preprocessing data
import librosa
import numpy as np

import os
import pathlib
import csv

import noisereduce as nr

from os import listdir
from os.path import isfile, join

import shutil

import warnings
warnings.filterwarnings('ignore')
####################################################################
#Load Files
####################################################################
print('Creating test splitted files...')
### Primeiramente para todo nosso conjunto de treino, vamos isolar os dados de cada classe

t_path = '../teste/'
onlyfiles = [f for f in listdir(t_path) if isfile(join(t_path, f))]

#Makes dirs
pathlib.Path(f'../teste/splitted').mkdir(parents=True, exist_ok=True)  

for en,file in enumerate(onlyfiles):
    data,fs = librosa.load(t_path+file)
    filename=file.split('.')[0]
    filename=filename.replace(' ','')
    #4 letras por audio
    for i in range(0,4):
        ini=int(i*(data.shape[0]/4))
        fim=int((i+1)*(data.shape[0]/4))
        #print(ini,fim)
        sample_data=data[ini:fim]
        librosa.output.write_wav(f'../teste/splitted/{filename}_{en}_{i}.wav',sample_data,int(data.shape[0]/4))
####################################################################
#Feature Extraction
####################################################################
print('Extract test features...')
header = 'filename chroma_stft chroma_cqt rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header = header.split()
file = open('../data/data_teste.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
classes = 'a,b,c,d,h,m,n,x,6,7'.split(',')
for filename in os.listdir(f'../teste/splitted/'):
    audioname = f'../teste/splitted/{filename}'
    y, sr = librosa.load(audioname, mono=True)
    #Noise Reduction considering last part only noise
    y = nr.reduce_noise(audio_clip=y,noise_clip=y[1750:],verbose=False)
    #Getting normalization
    y = librosa.util.normalize(y)
    
    #Chroma Frequencies
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    #print(chroma_stft)
    #Chroma cqt
    chroma_cqt = librosa.feature.chroma_cqt(y=y,sr=sr)
    #Aplying nearest neighbour filtering
    chroma_cqt = librosa.decompose.nn_filter(chroma_cqt,
                                      aggregate=np.median,
                                      metric='cosine')
    #Spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    #Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    #Spectral roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    #Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    #Mel-frequency cepstral coefficients (MFCC)(20 in number)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    #RMSE
    rmse = librosa.feature.rms(y=y)
    
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(chroma_cqt)} {np.mean(rmse)} \
    {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    #Expandindo o Mel-frequency em 20 atributos
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    file = open('../data/data_teste.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
#######################################################################################
shutil.rmtree('../teste/splitted/')
print('Teste extraido para "data_teste.csv"')
