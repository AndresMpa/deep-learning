# Recap - Red Neuronal Full Connect (FC)

$$
Z_j^l = \sum_{i=1}^{n}{w_{ji}^l} \space f^{l-1}(z_i) + b_j
$$

$$
Z_j^l = \sum_{i=1}^{n}{w_{ji}^l} \space a^{l-1}_i + b_j
$$

$$
a_i^{l-1} = f(z_j^l)
$$

> Este es el modelo Feed Forward

# Gradiente decendente

$$
W^{new} = W^{old} - \alpha \triangledown_w C
$$

$$
b^{new} = b^{old} - \alpha \triangledown_b C
$$

> Donde C es la función de perdida

$$
f(z_j^l + \triangle z_i^l)
$$

> Donde $\triangle$ es la acción del algoritmo de back propagation

$$
\triangledown_{zj}{C} = \frac{\partial C}{\partial z_j^l} \triangle z_j^l
$$

Algoritmo de back prograpagation

$$
\delta_j^l=\frac{\partial C}{\partial z_j^l}
$$

Usando la regla de la cadena

$$
\delta_j^l = \frac{\partial C}{\partial a_j^l} * \frac{\partial a_j^l}{\partial z_j^l}
$$

Pensando en la capa de salida

$$
l = L
$$

> Donde L es la última capa

$$
\delta_j^l = \frac{\partial C}{\partial a_j^L} \space \cancel{ \frac{\partial a_j^L}{\partial z_j^L}}
$$

$$
\cancel{ \frac{\partial a_j^L}{\partial z_j^L}} = f'(z_j^L)
$$

$$
\delta_j^L=\frac{\partial C}{\partial a_j^L} f'(z_j^L)
$$

La derivada parcial de $\partial z_j^l$ dependerá de la función de activación

> Es decir, sigmodal(z), Tanh(z), Lineal(z), Relu(z)...

Cuanod estamos en un proble de regresión

$$
C = \frac{1}{2} \sum^{k}_{i=1}{(y_i - a_i^L)^2}
$$

$$
\frac{\partial C}{\partial a_j^L} = \frac{1}{2} \sum^k_{i=1} \frac{\partial}{\partial a_j^L}(y_i - a_i^L)^2
$$

$$
\sum^k_{i=1}(y_i - a_i^L)\frac{\partial}{\partial a_i^L}(y_i-a_i^L)
$$

$$
= (y_j-a_j^L)(-1)
$$

$$
(a^L-y_j)
$$

Para regresión:

$$
\delta_j^L=(a_j^L-y_j)*f'(z_j^L)
$$

Función de perdida

$$
BCE= -\sum_{j=1}^{k}{ln(a_j^L) + (1-y_j) ln(1-a_j^L)}
$$

Para clasificar la probabilidad $a_i^L$

Para clasificación suele ser sigmoidal o softmax:

- Problemas biclase o binarios sigmoidal
- Problemas multiclase softmax

> Ambas para la capa $L$ la de salida

$$
\frac{\partial}{\partial a_j^L}BCE = \frac{-\partial}{\partial a_j^L}[y_j ln(a_j^L)+(1-y_j) ln(1-a_j^L)]
$$

$$
= -[y_j * \frac{1}{a_j^L} - \frac{1-y_j}{1-a_j}]
$$

$$
y_j \in [0, 1]
$$

Por lo que si $y_j = 0$

$$
\frac{\partial}{\partial a_j^L} BCE = \frac{1}{1-a_j^L}
$$

Si $y_j = 1$

$$
\frac{\partial}{\partial a_j^L} = \frac{-1}{a_j^L}
$$

# Resumne

Para determinar si el modelo está aprendiendo bien o no, podemos usar las funciones

> Nota: Estas funciones dan un valor, con el que se determina si el modelo está funcionando o no, pero se han de definir unos rangos por lado del investigador

## Si estamos en regresión

$$
\delta_j^L=(a_j^L - y_j)*f'(z_j^L)
$$

## Si estamos en clasificación

$$
\delta_j^L=\frac{1}{1-a_j^L}*f'(z_j^L)
$$

Ahora bien, para problemas biclase en las que usamos sigmoidal

$$
f(z_j^L) = \sigma(z_j^L)
$$

$$
f'(z_j^L) = \sigma(z_j^L)(1-\sigma(z_j^L))
$$

$$
\delta_l^L = \frac{1}{\cancel{1-\sigma(z_j^L)}}\sigma(z_j^L)\cancel{(1-\sigma(z_j^L))}
$$

Por lo que ser puede generalizar para los modelos que

$$
\delta_j^L = \sigma(z_j^L) = \sigma(z_j^L)-y_j
$$

En el modelo de step-descent estocastico, el parametro $m$ indica cuantas veces se ha de ejecutar el algoritmo