# Introducción a PyTorch

Torch es una framework open source creada para deep learning,
especificamente para deep learning, su calculo se basa en tensores
N-dimensionales; esencialmente los tensores serían los encargados
de almacenar las variables

> Los tensores se entienden como un tipo de calculo de matrices

> TensorFlow por su lado sería una opción a PyTorch

## Autodiferenciado automatico

Esto es una utilidad que viene en torch que permite derivar dada
una función, esto lo derivaria por si mismo

## Ventajas

- PyTorch define graficos computaciones dinámicos, en TensorFlow son estaticos; es decir, puede ir aprendiendo a medida que llegan los datos mientras que TensorFlow requiere tener todos los datos para iniciar

- La curva de aprendizaje de PyTorch es menos pesada que la de TensorFlow

> Para PyTorch el primer parametro a definir siempre ha de ser el mismo que la cantidad de entradas, estas se toman como una función lineal

```
# To do object dinamic mapping
 
z =  Model.named_paraneters()
print(z)
```

# Kernel

Es un modelo matematico, construido a partir de funciones bases; ejemplo las series de Fourier, la serie de Taylor, la sunficón sigmoidal,
la función tanh; es basicamente un producto entre 2 funciones bases, es un mapeo del espacio $X$ a $R^n$, eso es un kernel. Es un espacio de representación diferente en el que transformas de X a otro espacio más facil de medir. Transforma de un espacio de entrada a otro espacio que no es medible u observable.

El Kernel es un truco matematico que me permite tener diferentes representaciones; es como cuando se cambia de cordenadas cartesianas a cordenadas polares

Para DL, nos importan los kernels no parameticos o generativos; están más asociados con la distribución de los datos

> Matriz simetrica semidefinida: Aquella matriz cuyo determinante es mayor que 1