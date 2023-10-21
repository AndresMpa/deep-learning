# Red Neuronal Completamente conectada (Full connect / FC)

> MLP -> Multilayer perceptrón

## Resumen del modelo feed-forward

- Este es un modelo no lineal, diferente al modelo de perceptron
- Perceptron es una "hoja" que separa un plano
- FF es un modelo que te permite tener "curvas", pues este es un modelo no lineal

$$
z^{l} = w^{l} * a^{l -1} + b^{l}
$$

Para determinar especificamente el valor de una neurona se suele usar

$$
z_{j}^{l} = \sum_{i=1}^{n}{w_{ji}^{l} * a_{i}^{l -1}} + b_{i}^{l}
$$

> Donde $n$ es el número de neuronas de la capa $n$

$$
a_j^l=f(z_j^l)
$$

## Hiper-parametros del modelo

> Parametro: Cosas que necesitas para cumplir un objetivo

> Hiper-parametro: Son las condiciones, infraestructura que permiten al modelo entender

- $l$: Número de capas del modelo
- $a_{j}^{l}$: Función de activación
- $lr$: Learning rate

# Funciones de activación

- Lineal:

$$
lienal(z) = z
$$
$$
lienal'(z) = 1
$$

- Sigmoidal: Surge del teorma de byas y una función de distribución de probabilidad

$$
\sigma(z) = \frac{1}{1 + exp{-z}}
$$
$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

- Tangente hiperbolico:

$$
tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$

$$
tanh'(z) = 2\sigma(2z) - 1
$$

$$
tanh'(z) = 1 - tanh^{2}(z)
$$

- RELU:

$$
RELU = max(0, z)
$$

$$
RELU' = 1 \space si \space z \gt 0
$$

$$
RELU' = 0 \space si \space z \leq 0
$$