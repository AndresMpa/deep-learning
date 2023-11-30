# Redes neuronales recurrentes (RNN)

1) Modelo

$$
a^{(1)} = b + Wh^{(t-1)} + Ux^{(t)}
$$

$$
h^{(t)} = tanh(a^{(t)})
$$

$$
o^{(t)} = c + Vh^{(t)}
$$

$$
\hat{y}^{(t)} = softmax(o^{(t)})
$$

Suponiendo que en la salida hay varias etiquetas. Un problema multiclase

Para ese caso la estimación viene dada por

$$
\hat{y}^{(t)} = softmax(o^{(t)})
$$

Por tanto su loss (Función de perdida BCE) viene dado por:

$$
C^{(t)} = - \sum{y_i^{(t)}} log{\hat{y_i}^{(t)}} + (1 - y_i^{(t)}) log{(1 - \hat{y_i}^{(t)})} = L^{(t)} = L
$$

Los gradientes vendrás dados como:

$$
\triangledown_{o^{(t)}}L = \hat{y}_{j}^{(t)} - y_{j}^{(t)}
$$

$$
\triangledown_{h^{(\tau)}}L = (V)^T * \triangledown_{o^{(\tau)}}L^{(\tau)}
$$

> Donde $V$ es la matriz de $o^{(t)} = c + Vh^{(t)}$ y un vector compuesto por $\triangledown_{o^{(\tau)}}L^{(\tau)}$ donde cada termino viene dado por $\hat{y}_{j}^{(t)} - y_{j}^{(t)}$

$$
\triangledown_{h^{(t)}}L = (Diag(1 - (h^{(t + 1)})^{2})W)^{T} * \triangledown_{h^{(t+1)}}L + V^{T} \triangledown_{o^{(t)}}L
$$

# Long Short - Term Memory (LSTM)

Es una arquitectura muy común en aprendizaje de maquina.

Este modelo tiene 3 etapas:

- Compuerta de olvido: La olvido se encarga de determinar si un dato se debe guardar o no

- Conmpuerta de entrada: La de entrada es la encargada de ingresar un dato al modelo

- Compuerta de candidatos: La de candidato hace un operación matematica para determinar si que tan bueno es el dato nuevo

La caracteristica de este modelo implica que este modelo conserva información inicial

## Compuerta de olvido ($f_t$)

$$
f_t = f_{f}(b_{f}, x_t, h_{(t-1)}) = b_{f} + U_{xf} x_t + W_{hf} h_{(t-1)}
$$

Donde $f_f$ es una función de activación, que puede ser $lineal$, $sigmoidal$, $tanh$, etc.

## Compuerta de entrada ($i_t$)

$$
i_t = f_{i}(b_{i}, x_t, h_{t-1}) = b_{i} + U_{xi} x_t + W_{hi} h_{t-1}
$$

## Compuerta de candidato ($g_t$)

$$
g_t = f_{g}(b_g, x_t, h_{t-1}) = b_g + U_{xg} x_t + W_{hg} h_{t-1}
$$

## Estado de la celda $C_t$

$$
C_t = i_t \odot g_t + C_{t-1} \odot f_t
$$

## Salida de la compuerta $O_t$

$$
O_t = f_o(b_o + U_{xo} x_t + W_{ho} h_{t-1})
$$

## Salida de los estados $h_t$

$$
h_t = o_t \odot f(c_t)
$$

Si se ejecuta la iteración 1 de un problema aleatorio

```python
import numpy as np

if __name__ == '__main__':
    k = 0.01

    bf = np.array([
        [9.3],
        [5.8]])

    uxf = np.array([
        [3.60, 0.49, 0.12, 0.15],            
        [0.05, 0.19, 0.21, 0.19]])

    x1 = np.array([
        [1.0],
        [2.0],
        [0.0],
        [1.0]])

    whf = np.array([
        [4.9, 5.7],
        [9.0, 8.4]])

    ho = np.array([
        [3.0],
        [6.1]])

    result_for_ff_gate = k * bf + k * np.dot(uxf, x1) + np.dot(k * whf, k * ho) # f1
    
    print(result_for_ff_gate)
```