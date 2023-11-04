# Mejoramiento en el algoritmo de backpropagration

## Devanecimiento del gradiente (Gradient fading)

> Depende mucho de la función de perdida

Es te es un probrema inherente a las funciones de activación, en el cual hay un punto en el que el avance o aprendizajese

Cuando el modelo deja de aprender, una estrategia para evitar esto, es cambiar la función de perdida, por ejemplode LCE a BCE

Regularización: Es una tecnica para evitar que se sobre entre el modelo

El sobre entrenamiento tiene un comportamiento similar a un pulso cardiaco, de forma practico es como estudiante que estudia demasiado como hacer 1 ejercicio, por lo que uno nuevo con parametros diferentes, implicará que no tendría la capacidad de poder resolver un ejercicio nuevo

Cuando en mi modelo en test aumenta el error a medida que se va aumentando las epocas significa que el modelo está sobre entrenado

## Regularización

La idea es que los pesos no cambien tan rápido, para que generalice de manera más adecuada

Métodos para regularizar:

- Para antes de que se sobre entrene
- Que el modelo tenga una exactitud cercana al 100% no es necesariamente malo
- Accuracry entre test y trainig deberías se más o menos igual

Otra ventaja muy util es que se puede mejorar el performance y el rendimiento

El tamaño del modelo está condicionado por la cantidad de parametros

## Dropout

La idea es que la red sea dinamica, desconectas una parte de las neuronas en cada epoca del entrenamiento; puedes por ejemplo eliminar un porcentaje X de la red, luego se repite con otras neuronas que se seleccionarian de forma aleatorias y así, repitiendo este comportamiento cada vez

La idea de esta tecnica es que cada neurona de la red, generalize correctamente, esto hace que no se sobre entrene cada neurona. Se espera que al final el modelo que eso tenga no sea un modelo sobre entrenado

> Esto lo usan los transformer

Esta tenica hace samples más pequeños del modelo

> Tambien es importante ver el número de datos

> No se debe de tener más parametros que datos

Con este se terminan redes neuronales

La siguiente semana será redes convoluciones con PyTorch