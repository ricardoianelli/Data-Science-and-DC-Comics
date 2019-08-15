# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:17:17 2019

@author: RICARDO
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from sklearn.metrics import matthews_corrcoef

dc_database = pd.read_csv("dc-data.csv")

#Em qual ano apareceram os personagens mais antigos e quantos personagens apareceram naquele ano?
years = dc_database["YEAR"]
first_year = years.min()
older_characters = dc_database[dc_database["YEAR"] == first_year]

#Forneça os números principais da estatística descritiva para o campo que identifica a quantidade de aparições de uma personagem no formato:
#Média, Desvio Padrão, Mínimo, 25%, Mediana, 75%, Máximo.
dc_database["APPEARANCES"].describe()

#Qual a diferença entre o número de aparições entre heróis/heroínas e vilões/vilãs?
heroes = dc_database[dc_database["ALIGN"] == "Good Characters"]
villains = dc_database[dc_database["ALIGN"] == "Bad Characters"]
heroes_appear = heroes["APPEARANCES"].sum()
villains_appear = villains["APPEARANCES"].sum()
print(((heroes_appear-villains_appear)**2)**0.5)

#Em que ano apareceu o primeiro personagem bissexual? Quantas vezes esse personagem apareceu no total?
bissexuals = dc_database[dc_database["GSM"] == "Bisexual Characters"]
first_bissexual = bissexuals[bissexuals["YEAR"] == bissexuals["YEAR"].min()]
first_bissexual_appearances = first_bissexual["APPEARANCES"]

#Qual o valor da obliquidade da variável com o número de aparições de cada personagem? Ignore as casas decimais.
appearance_skewness = dc_database["APPEARANCES"].skew(axis=0, skipna=True)

#Qual o valor da curtose da variável com o número de aparições de cada personagem? Ignore as casas decimais.
appearance_kurtosis = dc_database["APPEARANCES"].kurtosis(axis=0, skipna=True)

#Com base nas informações sobre a distribuição da variável sobre aparições das personagens, como você classifica essa distribuição?
"""
a) Normal

b) Distribuição assimétrica á esquerda, com valores extremos além de 3 desvios padrões da média

c) Distribuição assimétrica á esquerda, com valores extremos abaixo de 3 desvios padrões da média

d) Distribuição assimétrica á direita, com valores extremos além de 3 desvios padrões da média

e) Distribuição assimétrica á direita, com valores extremos abaixo de 3 desvios padrões da média

#R: Since we have a positive skewness and a high positive kurtosis, it would be (b)
#Positive skewness indicates that most of the values are to the left of the mean.
#Positive kurtosis indicates high peaks on the values and extreme values above 3 std
"""

#Você deseja testar a hipótese que há uma tendência maior em reportar mulheres como heroínas do que homens como vilões.  
#Para isso, você quer testar se há de fato uma diferença significativa entre mulheres heroínas e homens vilões. Que tipo de teste é o mais adequado?
#R: X² test.

#Você deseja testar a hipótese que há uma tendência maior em reportar mulheres como heroínas do que homens como vilões.  
#Para isso, você quer testar se há de fato uma diferença significativa entre mulheres heroínas e homens vilões. 
#Baseado em um nível de confiança de 5%, utilize o teste mais adequado e responda:

#a) Qual o valor da estatística do teste (valor de Z ou T ou X2 ou F)?
#b) Qual o valor de p?
male_female = dc_database[np.logical_or(dc_database["SEX"] == "Female Characters", dc_database["SEX"] == "Male Characters")]
villains_heroes = male_female[np.logical_or(male_female["ALIGN"] == "Good Characters", male_female["ALIGN"] == "Bad Characters")]

freq_table = pd.crosstab(villains_heroes['SEX'], villains_heroes['ALIGN'])
print(freq_table)

chi2stat, p, dof, freq_exp = chi2_contingency(freq_table)
#Dof = (rows - 1) * (columns -1)
print('dof=%d' % dof)
print(freq_exp)

#c) A hipótese nula é rejeitada em favor da hipótese alternativa?   
prob = 0.95
#PPF = Percent point function
critical_value = chi2.ppf(prob, dof)
print(critical_value)

if abs(chi2stat) >= critical_value:
	print('Reject H0')
else:
	print('(Dont reject H0)')
    
#Para esta questão, considere apenas personagens bons (Good Characters) e maus (Bad Characters) 
# e identidades públicas (Public Identity) e secretas (Secret Identity). 
#Existe uma forte associação entre o alinhamento da personagem e o tipo de sua identidade? Qual o coeficiente de Matthews?

good_and_bad = dc_database[ np.logical_or(dc_database["ALIGN"] == "Good Characters", dc_database["ALIGN"] == "Bad Characters") ]
q10_data = good_and_bad[ np.logical_or(good_and_bad["ID"] == "Public Identity", good_and_bad["ID"] == "Secret Identity") ]

q10_cross = pd.crosstab(q10_data['ALIGN'], q10_data['ID'])
print(q10_cross)

q10_data = q10_data.dropna()
good = q10_data['ALIGN']=="Bad Characters"
public = q10_data['ID']=="Public Identity"

matthews_corrcoef(good, public)

#Quantos desvios padrões está da média uma personagem com 360 aparições? Use até duas casas decimais em sua resposta.
q11 = (360 - dc_database["APPEARANCES"].mean()) / dc_database["APPEARANCES"].std()
print(q11)

#Para esta questão, considere apenas personagens maus (Bad Characters) e identidades públicas (Public Identity) e secretas (Secret Identity). 
#Teste a hipótese de que personagens maus tendem a ser mais do tipo que possuem uma identidade secreta ao invés de uma identidade pública. Responda:
bad = dc_database[dc_database["ALIGN"] == "Bad Characters"]
q12_data = bad[ np.logical_or(good_and_bad["ID"] == "Public Identity", good_and_bad["ID"] == "Secret Identity") ]

#a) Qual o tipo de teste mais adequado para testar essa hipótese?
#R: Teste X²

#b) Qual o valor da estatística do teste (valor de Z ou T ou X2 ou F)?
q12_freq_table = pd.crosstab(q12_data['ID'], q12_data['ALIGN'])
print(q12_freq_table)

chi2stat, p, dof, freq_exp = chi2_contingency(q12_freq_table)
#Dof = (rows - 1) * (columns -1)
print('dof=%d' % dof)
print(freq_exp)
print(chi2stat)

#c) A hipótese nula é rejeitada em favor da hipótese alternativa?
prob = 0.95
#PPF = Percent point function
critical_value = chi2.ppf(prob, dof)
print(critical_value)

if abs(chi2stat) >= critical_value:
	print('Reject H0: There is no tendency to bad guys having a secret identity.')
else:
	print('Dont reject H0: There is, indeed, a tendency to bad guys having a secret identity.')