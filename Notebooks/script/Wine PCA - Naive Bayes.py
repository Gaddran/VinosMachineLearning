#!/usr/bin/env python
# coding: utf-8

# TODO:
# - Modificar markdowns para redacción en 3ra persona.
# - Finalizar la creación de nuevos markdowns.
# - Reformatear markdowns para tener presentación de informe.
# 
# IMPORTANTE: AL GENERAR EL PCA DEBEN CONSIDERARSE LAS VARIABLES PREDICTORAS, NO EL OBJETIVO (CLASS)

# <div>
# <img src="https://i.ibb.co/v3CvVz9/udd-short.png" width="150"/>
#     <br>
#     <strong>Universidad del Desarrollo</strong><br>
#     <em>Magíster en Data Science</em><br>
#     <em>Profesor: Tomás Fontecilla </em><br>
# 
# </div>
# 
# # Machine Learning
# *07 de Septiembre de 2024*
# 
# **Nombre Estudiante(s)**: `Gabriel Álvarez - Jaime Castillo - Kurt Castro - Giuseppe Lavarello`  

# #### Laboratorio 1
# 
# **Objetivo**: En este laboratorio deberá investigar del uso de la librería python scikit-learn y sus funciones más útiles. Su
# meta es realizar un análisis de datos, PCA y posterior clasficador utilizando el algoritmo Naive Bayes.
# Para esto, utilizará la base Wine que puede descargar desde 'https://archive.ics.uci.edu/ml/datasets/Wine' o bien utilizar
# la que se encuentra en la librería scikit-learn bajo el nombre de load_wine.
# 
# Se le evaluará por:
# 
# - Importe la libería Scikit-learn de forma correcta. Importe otras liberías que consideraría útiles para su análisis y
# comente por qué las usará. Cargue los datos.
# - Exploración de datos. Muestre información que considere relevante y explique brevemente por qué considera que lo es.
# - Selección de muestras. Explique cómo tomó las muestras para su análisis brevemente y qué hiperparámetros utilizó.
# - Ejecute el clasficador. Muestre sus resultados, argumente sobre el modelo. Argumente por qué hizo o dejó de hacer
# fine-tunning
# - Concluya.

# #### Carga de Librerías
# Las librerías que se utilizarán en este laboratorio son las siguientes:
# - pandas: Se utiliza la librería pandas para la manipulación de datos.
# - matplotlib: Se utiliza la librería matplotlib para la visualización de datos.
# - seaborn: Se utiliza la librería seaborn para generar gráficos específicos, como pairplot y heatmap.
# - StandardScaler de sklearn.preprocessing: Se utiliza para estandarizar los datos y no tener problemas con la escala de los mismos.
# - PCA de sklearn.decomposition: Se utiliza para determinar las componentes principales de los datos y reducir su dimensionalidad.
# - GaussianNB de sklearn.naive_bayes: Se utiliza para implementar el modelo de clasificación Naive Bayes.
# - train_test_split de sklearn.model_selection: Se utiliza para dividir los datos en conjuntos de entrenamiento y prueba.
# - precision_score, recall_score, f1_score, accuracy_score y confusion_matrix de sklearn.metrics: Se utilizan para evaluar el modelo de clasificación.

# Moví todos los import al principio. Es buena práctica porque permite mantener el código más ordenado y facilita la lectura.

# Para manipulación de datos 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Para visualización de datos
import seaborn as sns
import matplotlib.pyplot as plt

# Para clasificacion
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


# Realizaremos la carga de wine.csv y le asignaremos nombre a cada columna, ya que el dataset viene por defecto sin sus encabezados

wine = pd.read_csv("Wine.csv", header=None)

wine.columns = ["class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
                "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"]


# #### Exploración de Datos

# Vista general de la estructura del dataset

wine.head()


# Contaremos la cantidad de filas por cada categoría "class"

wine["class"].value_counts()


# Información general del dataset
wine.info()


# Información General del Dataset:
# - En total el dataset tiene 178 registros y 14 columnas.  
# - El dataset no contiene registros faltantes, por lo que no es necesario realizar imputación de datos.  
# - Todas las columnas del dataset son numéricas.

# Estadísticas básicas del dataset

wine.describe()


# A continuación, se mostrará un pairplot, para ver el comportamiento de las diferentes variables del dataset respecto a otra:

# sns.pairplot(wine, diag_kind="kde", hue="class", palette="pastel" ,markers=["o", "s", "D"])
sns.pairplot(wine, diag_kind='kde')
plt.show


# Del gráfico de variables cruzadas (pairplot) se puede observar que algunas variables pueden presentar relaciones lineales. Esto puede indicar que existe dependencia entre algunas variables. Por lo tanto, se puede reducir la dimensionalidad de los datos dejando las variable más significativas y que puedan explicar la mayor cantidad de varianza. Para ello se realiza un análisis de componentes principales (PCA) para determinar qué variables deben ser consideradas para el modelo.
# 
# Primeramente, se normaliza el dataset para que todas las variables tengan media 0 y varianza 1 en la variable **wine_standarized**. Esto es necesario para que todas las variables tengan el mismo peso al explicar la varianza y no sea dominada por variables con mayor escala.

wine_features = wine.drop("class", axis=1)

wine_standarized = StandardScaler(with_std=True,with_mean=True).fit_transform(wine_features)
pd.DataFrame(wine_standarized, columns = wine_features.columns).head()


# Para realizar el Análisis de Componentes Principales (PCA) se utilizará la clase PCA de la librería scikit-learn. Para ello, se aplica el método .fit_transform al set de datos estandarizado, se obtiene el ratio de varianza explicada por cada componente y luego se realiza un gráfico de la varianza explicada acumulada para determinar cuántos componentes se deben conservar.

# Se crea un objeto de la clase PCA
pca = PCA()

# Aplicar PCA a tus datos estandarizados
principal_components = pca.fit_transform(wine_standarized)

# Explorar la varianza explicada por cada componente principal
explained_variance_ratio = pca.explained_variance_ratio_

# Visualizar la varianza explicada acumulativa para decidir cuántos componentes conservar
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio.cumsum(), marker='o', linestyle='--')
plt.title('Varianza explicada acumulativa')
plt.xlabel('Número de componentes principales')
plt.ylabel('Varianza explicada acumulativa')
plt.show()


# Como se observa en el gráfico anterior, la varianza es explicada aproximadamente en un 40% por el primer componente, mientras que cerca de un 70% de esta es explicada por los tres primeros. Para este caso, para evitar una gran pérdida de información al realizar una reducción de dimensionalidad, se conservarán ocho componentes, ya que estos aportan a la explicación de aproximadamente un 92.02% de la varianza.

n_components = 8
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(wine_standarized)

# Ahora tienes tus datos reducidos a n_components características
pc_df=pd.DataFrame(principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])
print(f'Porcentaje de varianza explicada por los primeros {n_components} componentes principales: {pca.explained_variance_ratio_.sum()*100:.2f}%')


# Se realiza un nuevo pairplot para visualizar el comportamiento de los componentes principales entre sí.

sns.pairplot(pc_df, diag_kind="kde")
plt.show


# Del gráfico anterior podemos visualizar a simple vista que no existen relaciones lineales entre los componentes principales elegidos.
# 
# A continuación, se realiza un análisis de la composición de los componentes principales generados para poder determinar cuánto aporta cada variable a cada uno de ellos.

# Crear un DataFrame con las cargas de cada variable en cada componente principal
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(1, n_components + 1)], index=wine_features.columns)

#Importancia de las variables
loadings_abs = loadings.abs()

# Mostrar las 5 variables con las cargas más altas para cada componente principal
top_loadings = loadings_abs.apply(lambda x: x.nlargest(5).index)

# Mostrar las cargas
print("Cargas de variables en cada componente principal:")
print(top_loadings)


# Para un mejor entendimiento, se generan gráficos de barras horizontales para visualizar el aporte de cada variable a los componentes principales.

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


# # Todo este análisis tiene que cambiar porque quedó fuera la variable "class"
# - **Componente Principal 1**:
# 
# Las variables más influyentes son class, flavanoids, total_phenols, od280/od315_of_diluted_wines, y proanthocyanins.
# Estas variables tienen una alta carga absoluta, lo que indica que tienen una gran influencia en la definición de este primer componente.
# - **Componente Principal 2**:
# 
# Las variables color_intensity, alcohol, proline, ash, y magnesium son las más influyentes.
# Estas características contribuyen significativamente a la varianza explicada por el segundo componente principal, indicando que este componente está capturando variabilidad relacionada con la intensidad de color y composición química de las muestras.
# 
# - **Componente Principal 3**:
# 
# Las variables principales aquí son color_intensity, alcalinity_of_ash, alcohol, nonflavanoid_phenols, y od280/od315_of_diluted_wines.
# Este componente parece estar relacionado con aspectos químicos específicos y propiedades del color.
# 
# - **Componente Principal 4**:
# 
# Las variables destacadas son malic_acid, proanthocyanins, hue, nonflavanoid_phenols, y proline.
# Este componente puede estar capturando variabilidad relacionada con la acidez y coloración de las muestras.
# 
# - **Componente Principal 5**:
# 
# Las variables más influyentes son magnesium, nonflavanoid_phenols, alcohol, malic_acid, y class.
# Este componente podría estar enfocado en la variabilidad relacionada con la composición mineral y la acidez.
# 
# - **Componente Principal 6**:
# 
# Las variables claves son malic_acid, proanthocyanins, color_intensity, nonflavanoid_phenols, y od280/od315_of_diluted_wines.
# Este componente parece capturar variabilidad relacionada con la acidez, la intensidad de color y la composición fenólica.
# 
# - **Componente Principal 7**:
# 
# Las variables principales son nonflavanoid_phenols, proanthocyanins, magnesium, alcalinity_of_ash, y malic_acid.
# Este componente está relacionado con variabilidad en la composición mineral y fenólica.
# 
# - **Componente Principal 8**:
# 
# Las variables más influyentes son hue, alcohol, alcalinity_of_ash, total_phenols, y proanthocyanins.
# Este componente puede estar relacionado con variabilidad en la coloración y la composición química.

# Ahora importaremos el modelo de clasificación GaussianNB del módulo Naive Bayes de Scikit-Learn

# Creamos la instancia para implementar algoritmo de Naive Bayes
model = GaussianNB()


# Defino cual es mi variable objetivo dentro del dataset

wine_type= wine["class"]


# A continuación, usaremos la función train_test_split para separar nuestros datos en un conjunto para entrenar el modelo. La proporción a utilizar será de 70/30.
# Con esto ya hemos realizado una selección de muestra y definido hiperparámetros

# Dividiremos nuestros datos en dos subconjuntos: uno para entrenamiento (train) y otro para pruebas (test). Dado lo anterior importaremos la siguiente función:

X_train, X_test, y_train, y_test = train_test_split(pc_df, wine_type, train_size = 0.7, random_state=1234)


model.fit(X_train,y_train)


# Crearemos la variable "labels", la cual almacenará las etiquetas del modelo para el conjunto de datos de prueba X_test

labels = model.predict(X_test)


# Ahora realizaremos una matriz de confusión, con la cuál podremos analizar los elementos que se muestren en la diagonal, que serán las instancias que sean correctamente clasificadas, y fuera de la diagonal, errores que existan en la clasificación:

mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels= ["class 1", "class 2", "class 3"], yticklabels= ["class 1", "class 2", "class 3"])
plt.xlabel('True Label')
plt.ylabel('Predicted Label');


# 
# - (1,1): 16 - El modelo predijo correctamente 16 veces la primera clase (clase 1).
# - (2,2): 21 - El modelo predijo correctamente 21 veces la segunda clase (clase 2).
# - (3,3): 16 - El modelo predijo correctamente 16 veces la tercera clase (clase 3).
# 
# - (2,3): 1 - El modelo predijo una vez la segunda clase (clase 2), cuando la etiqueta verdadera era la tercera clase (clase 3). Este es un error del modelo.
# 
# El modelo muestra en general un muy buen desempeño. Esto se podrá evidenciar una vez calculemos las métricas para evaluar el rendimiento del modelo, lo que se presentará a continuación:

print(f'precision: {precision_score(labels, y_test, average="macro")}')
print(f'recall: {recall_score(labels, y_test, average="macro")}')
print(f'accuracy: {accuracy_score(labels, y_test)}')
print(f'f1 score: {f1_score(labels, y_test, average="macro")}')


# ##### **Conclusión**
# 
#  El modelo tiene una alta precisión, lo que significa que casi todas las predicciones positivas fueron correctas, y un buen recall, lo que indica que identificó correctamente la mayoría de las instancias positivas.
# 
#  Al observar los resultados de las 4 métricas, podemos concluir que se presenta un modelo con un gran desempeño y es confiable para lo que fue entrenado. Es por esto que no es necsario realizar ajustes o fine-tuning al modelo.
