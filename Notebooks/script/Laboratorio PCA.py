#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


vinos = pd.DataFrame(load_wine().data,columns = load_wine().feature_names)


# # Analisis Exploratorio

vinos.head()


vinos.shape


vinos.info()


vinos.describe()


# Visualización de componentes principales en pares
sns.pairplot(vinos, diag_kind='kde', corner=True)  # 'diag_kind' permite mostrar histogramas de densidad en la diagonal principal
plt.show()


# ## Estandarizar variables
# 

X = StandardScaler(with_std=True,with_mean=True).fit_transform(vinos)
pd.DataFrame(X,columns = load_wine().feature_names).head()


# Crear una instancia de PCA
pca = PCA()

# Aplicar PCA a tus datos estandarizados
principal_components = pca.fit_transform(X)

# Explorar la varianza explicada por cada componente principal
explained_variance_ratio = pca.explained_variance_ratio_

# Visualizar la varianza explicada acumulativa para decidir cuántos componentes conservar
import matplotlib.pyplot as plt

plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio.cumsum(), marker='o', linestyle='--')
plt.title('Varianza explicada acumulativa')
plt.xlabel('Número de componentes principales')
plt.ylabel('Varianza explicada acumulativa')
plt.axvline(x=7, ls="--", color='r')
plt.show()


# # Hasta aca llegue

n_components = 7
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X)

# Ahora tienes tus datos reducidos a n_components características
pc_df=pd.DataFrame(principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])


pca.explained_variance_ratio_[0:(n_components-1)].sum()


# Visualización de componentes principales en pares
sns.pairplot(pc_df, diag_kind='kde')  # 'diag_kind' permite mostrar histogramas de densidad en la diagonal principal
plt.show()


# # Hasta aca llegue
# ----

# # Importancia de las variables en cada componente
# La elección del número de variables que debemos retener de los componentes principales depende de los objetivos de análisis y de la cantidad de información que estés dispuesto a perder en el proceso de reducción de dimensionalidad.
# 

import pandas as pd

# Crear un DataFrame con las cargas de cada variable en cada componente principal
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(1, n_components + 1)], index=vinos.columns)

#Importancia de las variables
loadings_abs = loadings.abs()

# Mostrar las 5 variables con las cargas más altas para cada componente principal
top_loadings = loadings_abs.apply(lambda x: x.nlargest(5).index)

# Mostrar las cargas
print("Cargas de variables en cada componente principal:")
print(top_loadings)


# ###  Crear gráficos de barras para las variables más correlacionadas con cada componente principal
# Las cargas son los coeficientes que indican la correlación o la contribución de cada variable original a un componente principal.
# 
# - Cargas cercanas a 0: Las variables con cargas cercanas a 0 tienen una contribución mínima al componente principal correspondiente y, por lo tanto, tienen poco impacto en la estructura del componente. Puedes considerar eliminar o no darles mucha importancia si deseas simplificar la interpretación del componente.
# 
# - Cargas cercanas a 0.6: Las variables con cargas cercanas a 0.6 tienen una contribución significativa al componente principal. Esto significa que estas variables tienen un fuerte impacto en la formación del componente y son importantes para entender la estructura de los datos en ese componente.
# 
# - Cargas intermedias: Las cargas que se encuentran entre 0 y 0.6 indican una contribución moderada de la variable al componente principal. Estas variables son relevantes pero pueden no ser tan dominantes como las que tienen cargas más altas. La interpretación de estas cargas dependerá de tu objetivo de análisis y de cómo deseas utilizar los componentes principales.

plt.figure(figsize=(15, 10))

for i in range(n_components):
    plt.subplot(3, 4, i + 1) #grafico de 3 filas y 3 columnas para visualizar cada componente
    top_vars = top_loadings[f'PC{i+1}'] #muestra los variables con mayor importancia para cada componente
    loading_values = loadings_abs[f'PC{i+1}'][top_vars]
    plt.barh(top_vars, loading_values, color='skyblue')
    plt.xlabel('Carga Absoluta')
    plt.title(f'Componente Principal {i+1}')

plt.tight_layout()
plt.show()


# ##### Nombre de componentes
# 
# Componente Principal 1 (PC1): "Características de Contorno y Dimensiones"
# 
# Componente Principal 2 (PC2): "Propiedades de Superficie y Textura"
# 
# Componente Principal 3 (PC3): "Errores de Medición y Simetría"
# 
# Componente Principal 4 (PC4): "Características de Composición y Suavidad"
# 
# Componente Principal 5 (PC5): "Errores de Textura y Simetría"
# 
# Componente Principal 6 (PC6): "Propiedades de Suavidad y Dimensión Fractal"
# 
# Componente Principal 7 (PC7): "Características de Dimensión Fractal y Puntos de Contorno"
# 
# 

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


cancer_type= load_breast_cancer().target


X_train, X_test, y_train, y_test = train_test_split(pc_df, cancer_type, train_size = 0.7, random_state=1234)


model.fit(X_train,y_train)


labels = model.predict(X_test)


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=load_breast_cancer().target_names, yticklabels=load_breast_cancer().target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');


from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score


print(f'precision: {precision_score(labels,y_test)}')
print(f'recall: {recall_score(labels,y_test)}')
print(f'accuracy: {accuracy_score(labels,y_test)}')
print(f'f1 score: {f1_score(labels,y_test)}')

