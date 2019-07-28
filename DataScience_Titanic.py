# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:36:49 2019

@author: mateus
"""
# imports de bibliotecas
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer

#%%
# import base de dados de treino
base_treino = pd.read_csv('train.csv')

base_teste = pd.read_csv('test.csv')

#%%
# analise exploratoria

base_treino.info()
# idade, cabin e embarked com campos em branco

#%%
# remover colunas que não serão utilizadas como nome, id e cabin

base_treino.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#%% base de test
base_teste.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
#%%
base_teste.info()

#%%
# media das idades
media_idade = base_treino['Age'].mean()
#%%
media_idade_test = base_teste['Age'].mean()

#%%
# pre-processamento para resolver dados faltantes na coluna idade
# sera atribuida a media para os campos faltantes
base_treino['Age'].fillna(media_idade, inplace=True)

#Imputer(missing_values=np.nan, strategy = 'mean') terminar isso depois
#%%
base_treino['Embarked'].fillna(base_teste['Embarked'].value_counts().idxmax() , inplace=True)

#%%
base_teste['Age'].fillna(media_idade_test, inplace=True)
#%%
base_teste['Fare'].fillna(base_teste['Fare'].mean() , inplace=True)
#%%
# separar base em previsores e classe
classe = base_treino.iloc[:, 0:1] # não usar o .values pois senão a coleção vira do tipo object que são possue os comandas do pandas como info ou describe

previsores = base_treino.iloc[:, 1:8]

#%% 

previsores.info()

#%% 
# vizualizar o tipo de dado de cada coluna
previsores.dtypes

#%%
# transformar um tipo de dado object em string
previsores['Embarked'] = previsores['Embarked'].astype('str') 

#%%
# aplicar onehotencoder

label_encoder = LabelEncoder()

previsores['Embarked']  = label_encoder.fit_transform(previsores['Embarked'])
#%%

previsores['Sex']  = label_encoder.fit_transform(previsores['Sex'])

#%%
base_teste['Sex'] = label_encoder.fit_transform(base_teste['Sex'])
base_teste['Embarked'] = label_encoder.fit_transform(base_teste['Embarked'])

#%%
# preencher dados faltantes do campo embarked

#imputer = Imputer(missing_values=np.nan, strategy = 'most_frequent')
#imputer = imputer.fit(previsores['Embarked'])
#previsores['Embarked'] = imputer.transform(previsores['Embarked'])


#%%

# aplicar transformação dos dados descritivos e escalonar
# é passado como parametro o indice das colunas que sera transformado
one_hot_encoder = OneHotEncoder(categorical_features = [1, 6])

#%%

previsores_hotEncoder = previsores
previsores_hotEncoder = one_hot_encoder.fit_transform(previsores).toarray()

#%%

#one_hot_encoder_test = OneHotEncoder(categorical_features = [1, 6])    

base_teste_hotEcoder = one_hot_encoder.fit_transform(base_teste).toarray()

#%%
# não esta usado
scaler = StandardScaler()
previsores[:, :] = scaler.fit_transform(previsores[:, :])


#%%
# declarar o modelo
classificador = MLPClassifier(verbose=True,
                              max_iter=1000,
                              tol=0.000010)

#%%

#random forest

from sklearn.ensemble import RandomForestClassifier

# classificador_rf = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)

classificador_rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

classificador_rf.fit(previsores, classe)

previsoes_rf = classificador_rf.predict(base_teste)

#%%
# treinar o modelo
classificador.fit(previsores, classe)   
    
#%%
#fazer previsoes
previsoes = classificador.predict(base_teste)

#%%

classe_teste = pd.read_csv('gender_submission.csv')
#%%

classe_teste.drop(['PassengerId'], axis=1, inplace=True)

#%%
# precisao e matriz de comfusao RNA

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)


#%%
# precisao e matriz de comfusao RF
precisao = accuracy_score(classe_teste, previsoes_rf)
matriz = confusion_matrix(classe_teste, previsoes_rf)