# Introducción a la convolución y operaciones convulocionadas

Una convolución es una permutación de una señal original que se puede recuperar, esto genera una nueva representación de una señal, la convolución surge de una integrar interatable, usualmente se trata convoluciones en 1 dimensión pero se pueden usar en multiples dimensiones.

A efectros practicos, la convolución sirve a manera de "filtro" para representar una información de una forma diferente (Como un zip), esto para poder recoger datos en un plano de representación diferente, dependiendo del kernel que se use se puede reducir o aumentar la complejidad de un conjunto de datos

En una red convolucional lo que se quiere es encontra los valores del kernel, se sintonizan los parametros del kernel usando backpropagation

# Nomenclatura

- $i_{1,2}$ dimensión de la matriz de entrada (Tensor)
- $k_{1,2}$ dimensión del kernel
- $S_{1,2}$ (Stripe) que tanto se va a mover el kernel en $i$ y $j$ (Se ve facilmente posicionando el kernel en la mitad de la matriz)
- $p_{1,2}$ (Padding) Cantidad de 0 agregados dado un kernel más grande que la matriz (tensor) de entrada