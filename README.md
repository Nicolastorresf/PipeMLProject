# Predicción de Ruido de Perfiles Aerodinámicos (NASA Airfoil) con PySpark ML Pipelines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ## Descripción General

Este proyecto implementa un pipeline completo de Machine Learning utilizando **Apache Spark (PySpark)** para predecir el nivel de presión sonora (en decibelios) generado por perfiles aerodinámicos (airfoils), basándose en el conjunto de datos "NASA airfoil self-noise". El flujo de trabajo abarca desde la limpieza y preparación de los datos (ETL) hasta la creación, evaluación y persistencia de un modelo de regresión lineal.

El objetivo es demostrar un enfoque práctico y escalable para resolver problemas de regresión mediante pipelines de `pyspark.ml` en un entorno como Google Colab.

## Conjunto de Datos

* **Fuente:** [NASA airfoil self-noise Data Set](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) (UCI Machine Learning Repository).
* **Licencia del Dataset:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Descarga:** El script descarga automáticamente la versión modificada (`NASA_airfoil_noise_raw.csv`) proporcionada para este proyecto.
* **Características (Features):**
    * `Frequency` (integer): Frecuencia, en Hertzs.
    * `AngleOfAttack` (double): Ángulo de ataque, en grados.
    * `ChordLength` (double): Longitud de la cuerda, en metros.
    * `FreeStreamVelocity` (double): Velocidad de flujo libre, en metros por segundo.
    * `SuctionSideDisplacement` (double): Espesor de desplazamiento del lado de succión, en metros.
* **Variable Objetivo (Target):**
    * `SoundLevelDecibels` (double): Nivel de presión sonora escalado, en decibelios (originalmente `SoundLevel`).

## Metodología y Flujo de Trabajo

El proyecto sigue una estructura en cuatro partes:

1.  **Parte 1: ETL (Extracción, Transformación y Carga)**
    * Carga del dataset CSV crudo en un Spark DataFrame.
    * Limpieza inicial: Se eliminan 19 filas duplicadas (`1522` -> `1503` filas).
    * Manejo de nulos: Se eliminan filas con valores nulos (`1503` -> `1499` filas). *(Nota: En esta ejecución, no se encontraron nulos después de eliminar duplicados)*.
    * Transformación: Se renombra la columna `SoundLevel` a `SoundLevelDecibels`.
    * Almacenamiento: El DataFrame limpio se guarda en formato **Parquet** (`NASA_airfoil_noise_cleaned.parquet`) para optimizar lecturas posteriores.

2.  **Parte 2: Creación del Pipeline de Machine Learning**
    * Se carga el DataFrame limpio desde Parquet (`1499` filas).
    * Se define un pipeline de `pyspark.ml` con **3 etapas**:
        1.  `VectorAssembler`: Agrupa las 5 columnas de características en un único vector llamado `features`.
        2.  `StandardScaler`: Escala el vector `features` para obtener `scaledFeatures`, lo cual ayuda al rendimiento de la regresión lineal.
        3.  `LinearRegression`: Configura el modelo para predecir `SoundLevelDecibels` a partir de `scaledFeatures`.
    * Los datos se dividen en conjuntos de **entrenamiento (70% - 1101 filas)** y **prueba (30% - 398 filas)** usando una semilla fija (`seed=42`) para reproducibilidad.
    * Se entrena el pipeline completo con los datos de entrenamiento (`pipeline.fit()`).

3.  **Parte 3: Evaluación del Modelo**
    * Se generan predicciones (`prediction`) sobre el conjunto de datos de prueba usando el pipeline entrenado (`pipelineModel.transform()`).
    * Se evalúa el rendimiento del modelo de Regresión Lineal usando `RegressionEvaluator`:
        * **MSE (Mean Squared Error):** `25.00`
        * **MAE (Mean Absolute Error):** `3.91` decibelios
        * **R² (R-Squared):** `0.50`

4.  **Parte 4: Persistencia del Modelo**
    * El pipeline entrenado (`pipelineModel`) se guarda en disco en la ruta `Final_Project_Airfoil_Model`.
    * El modelo guardado se carga de nuevo (`PipelineModel.load()`).
    * Se verifica que el modelo cargado funciona correctamente realizando predicciones sobre los datos de prueba.

## Resultados y Conclusiones

* **Limpieza de Datos:** El proceso ETL fue efectivo, identificando y eliminando duplicados, resultando en un dataset final de 1499 registros listos para el modelado.
* **Rendimiento del Modelo:** El modelo de Regresión Lineal, integrado en el pipeline de PySpark, alcanzó un **R² de 0.50** en el conjunto de prueba. Esto indica que el modelo explica aproximadamente el **50% de la variabilidad** en el nivel de presión sonora. El **Error Absoluto Medio (MAE) fue de 3.91 dB**, lo que sugiere que, en promedio, las predicciones del modelo se desvían unos 3.9 decibelios del valor real. Si bien el modelo tiene capacidad predictiva, un R² de 0.50 sugiere que hay una cantidad significativa de varianza no explicada, y podría explorarse el uso de modelos más complejos (ej., Gradient Boosting, Random Forest) o ingeniería de características adicional para mejorar la precisión si fuera necesario.
* **Análisis de Coeficientes:** Los coeficientes del modelo de regresión lineal (aplicados a las características *escaladas*) fueron:
    * `Frequency`: -3.9906
    * `AngleOfAttack`: -2.2881
    * `ChordLength`: -3.3269
    * `FreeStreamVelocity`: 1.4832
    * `SuctionSideDisplacement`: -2.0551
    Esto sugiere que, manteniendo las demás constantes, un aumento en `Frequency`, `AngleOfAttack`, `ChordLength`, y `SuctionSideDisplacement` tiende a *disminuir* el nivel de sonido predicho, mientras que un aumento en `FreeStreamVelocity` tiende a *aumentarlo*. La magnitud indica la fuerza relativa de estas relaciones *para los datos escalados*.
* **Persistencia:** El modelo se guardó y cargó con éxito, demostrando la capacidad de reutilizar el pipeline entrenado sin necesidad de reentrenamiento, lo cual es crucial para aplicaciones prácticas.

En general, este proyecto ilustra un ciclo de vida completo de Machine Learning utilizando PySpark, desde la preparación de datos hasta la evaluación y persistencia del modelo, proporcionando una base sólida para abordar problemas de regresión con datos tabulares.

## Tecnologías Utilizadas

* **Python**
* **Apache Spark (PySpark 3.5.0):**
    * Spark SQL (DataFrames)
    * MLlib (Pipelines, VectorAssembler, StandardScaler, LinearRegression, RegressionEvaluator)
* **Google Colab:** (Entorno de ejecución del notebook)

## Configuración y Uso

1.  **Entorno:** Abre el archivo `.ipynb` en Google Colab o un entorno con PySpark 3.5 instalado.
2.  **Instalación:** Ejecuta la primera celda para instalar `pyspark` si es necesario:
    ```bash
    !pip install pyspark==3.5 -q
    ```
3.  **Ejecución:** Ejecuta todas las celdas del notebook secuencialmente. El script descargará los datos necesarios, realizará el ETL, entrenará el modelo, lo evaluará y lo guardará.

## Estructura de Archivos (Ejemplo)
