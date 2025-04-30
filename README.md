# PipeMLProject
## Conclusiones del Proyecto

Este proyecto demostró exitosamente la construcción de un flujo de trabajo completo de Machine Learning utilizando PySpark para predecir el nivel de presión sonora (`SoundLevelDecibels`) del conjunto de datos "NASA airfoil self-noise".

Los objetivos clave se cumplieron de la siguiente manera:

1.  **ETL Efectivo:** El proceso de Extracción, Transformación y Carga (ETL) se realizó satisfactoriamente. Se cargó el dataset inicial (`[rowcount1]` filas), se eliminaron las filas duplicadas (`[rowcount2]` filas restantes) y las filas con valores nulos, resultando en un conjunto de datos limpio y consistente de `[rowcount3]` filas. Este dataset depurado se almacenó eficientemente en formato Parquet.

2.  **Pipeline de ML Funcional:** Se diseñó y ejecutó un pipeline de `pyspark.ml` que incluyó las etapas necesarias de preprocesamiento (`VectorAssembler` para unificar características, `StandardScaler` para escalarlas) y modelado (`LinearRegression`). Este enfoque modular facilita la reproducibilidad y el manejo del flujo de trabajo.

3.  **Rendimiento del Modelo:** El modelo de Regresión Lineal entrenado sobre los datos escalados demostró ser capaz de predecir el nivel de ruido. Las métricas de evaluación clave fueron:
    * **R² (R-cuadrado):** Aproximadamente `[Valor R²]` (Ej: 0.55), indicando que el modelo explica alrededor del `[Valor R² * 100]`% de la variabilidad en el nivel de sonido.
    * **MAE (Error Absoluto Medio):** Aproximadamente `[Valor MAE]` (Ej: 3.5) decibelios, representando el error promedio de predicción.
    * **MSE (Error Cuadrático Medio):** `[Valor MSE]`.
    *(Comentario Opcional: Puedes añadir una breve interpretación aquí, por ejemplo: "Aunque el R² indica una capacidad predictiva moderada/razonable, el MAE sugiere que las predicciones se desvían en promedio X decibelios del valor real, lo que podría ser aceptable/mejorable dependiendo de la aplicación.")*

4.  **Análisis de Características:** El modelo de regresión lineal proporcionó coeficientes para cada característica de entrada (escalada). El análisis de estos coeficientes *(ej: `print(f"  Coefficient for {i} is {round(j, 4)}")`)* sugiere que variables como `[Nombre Característica 1]` y `[Nombre Característica 2]` tienen una influencia [positiva/negativa] significativa en la predicción del nivel de sonido. *(Adapta esto según los coeficientes que observaste)*.

5.  **Persistencia Exitosa:** El pipeline completo entrenado (`pipelineModel`) se guardó correctamente en disco y se pudo volver a cargar y utilizar para realizar predicciones, validando su funcionalidad para un posible despliegue o uso futuro.

En resumen, este proyecto no solo cumplió con los requisitos de limpieza de datos, construcción de pipeline, evaluación y persistencia, sino que también proporcionó una base sólida en la aplicación de PySpark para resolver problemas de regresión en un contexto práctico de ingeniería aeronáutica.
