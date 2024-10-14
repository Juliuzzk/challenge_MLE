# Software Engineer (ML & LLMs) Challenge

## Selección del modelo

Luego de realizar un análisis e investigación de los modelos presentados en `exploration.pynb`, se tomo la decisión de usar el modelo `XGBoost with Feature Importance and with Balance` esto debido a que este modelo tiene varias ventajas sobre el resto de modelos presentados, resaltando las sigueintes:

1. **Mejora la precisión**: XGBoost es un algoritmo poderoso que maneja bien datos desbalanceados, y el uso de técnicas de balanceo como SMOTE o ajuste de pesos mejora la precisión en clases minoritarias.
2. **Identificación de características clave**: La importancia de características (feature importance) permite identificar qué variables tienen mayor impacto en las predicciones, ayudando a optimizar el modelo y reducir el overfitting.
3. **Manejo eficiente de datos desbalanceados**: XGBoost maneja pesos en las clases, lo que, combinado con un conjunto de datos balanceado, permite que el modelo sea más robusto y generalice mejor en datasets desequilibrados.
4. **Rendimiento y rapidez**: XGBoost es conocido por ser rápido y eficiente en el uso de memoria, acelerando el proceso de entrenamiento y predicción.

## Algunos errores encontrados

- Las llamadas a `sns.barplot` en `exploration.pynb` no incluía la definición de los parámetros `x=` y `y=`, por lo cual, se agrega esta variable en todas las llamadas de barplot para poder realizar su correcta ejecución.
- La librería `xgboost` no se encontraba especificada en los archivos `requirements*.txt`, por lo cual, se agrega en el archivo correspondiente.
- Durante la ejecución se encontró un problema con la versión de la librería `numpy`, por lo cual, se realizo un downgrade a la versión `1.23.5`
- Al ejecutas los test del `Makefile`, se encontró un problema con la definición de la ruta de `data.csv`, por lo cual, se actualiza la ruta para garantizar la lectura del archivo.
- Se realiza una corrección en la funciona `is_high_season`, ya que este no estaba considerando la hora en los intervalos de comparación.
