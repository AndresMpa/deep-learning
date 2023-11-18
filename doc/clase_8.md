## Pooling

Es un tipo de operación que disminuye la dimensionalidad del tensor de entrada, lo que garantiza esto es que el modelo no sea sensible a cambios en la entrada, por ejemplo tomando los valores maximos

## Convolución transpuesta (Desconvolución)

Esto es similar a descargar un archivo .part se tiene la data pero desordenada y esta se recostruye de una forma un poco diferente no será necesariamente igual pero si parecido, esto es como ir hacia atras en donde la convolución es ir hacia adelante

#### Nomenclatura

- $i'$ Tamaño de la matriz convolucionada
- $i$ Tamaño de l amatriz anterior
- $s'$ (Strip) de la matriz convolucionada
- $s$ (Stripe) de la matriz anterior
- $p'$ (Padding) de la matriz convolcionada
- $p$ (Padding) de la matriz anterior
- $k$ tamaño del kernel

> La desconvolución se puede ver como un gradiente, por tanto se estaría encontrando un gradiente a la matriz de entrada

## Otras convoluciones

- Convoluciones dilatada: Sirve para eliminar operaciones, por ejemplo para aumentar la resolución de una imagen
- Convolución no lineal: Esta compuesta por una parte lineal y una no lineal. Muy pocos fenomenos son no lineales, sirve para tener una mejor representación de los datos. Debe intentarse en lo maximo posible de NO usarse, pues aumenta mucho el peso del calculo
- Convoluciones en 2D y 3D: Esto es basicamente una proyección que sintetiza a una nueva representación o cosa