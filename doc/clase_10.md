# Review

- Perceptrón (Modelo lineal simple): Usa matrices o vectores
- Perceptrón multicapa o Neural Network o Full Connect Neural Network (No lineal - Multiplicación matricial): Usa matrices o vectores
- Redes convolucionadas - CNN (No lineal - Convolución): Usa tensores 

## Redes recurrentes - RNN (Secuencias de estados)

Estado: Es una instacia de una secuencia de valores, es como un trozo de un circulo infinito

> Para regresar se necesita de funciones inversas

Se define como:

$$
h^{(t)} = f(h^{(t-1)}, x^{(t)}, \theta)
$$

$h^{(t-1)}$: Estado en el instante $t$
$x^{(t)}$: Entrada en el instante $t$
$\theta$: Parámetros del modelo

Este tipo de redes sirven para cualquier cosa que requiera o tenga secuencias

### Notación

- $\tau$: Instante final de la secuencia
- $h_t$, $h^t$, $h^{(t)}$: Estado en el instante de tiempo $t$
- $o$: "Output" salida estimada
- $y$: Salida esperada o valor real
- $L$: Loss o perdida del modelo
- $x$: Parametro externo
- $W$, $U$, $V$: Son matrices que se actualizan, basicamente como la matriz de pesos weight

Generalmente f1 se un tangente hiperbolico, ya que normaliza en -2 y 2

Mientras $f_{out}$ dependenrá de la tarea

> Lienal: Para regresión 
> Sigmoide: Para clasificación de 1 clases
> Softwax: Para clasificación de multiples clases

> BCE: En la perdida en problemas de clasificación
> LSE: Es la perdida en problemas de regresión

Si se quiere entrar el modelo dada la secuencia $x^{(t)}$ y $y^{(t)}$ ¿Qué debo determinar?

Respuesta: $U$, $V$, $W$, $b$ & $C$

¿A que corresponden los parametros del modelo?

Para entrenar el modelo se aplica: Backpropagation