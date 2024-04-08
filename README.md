# Formulario de Examen

Este es un formulario de examen para repasar los conceptos importantes. Aquí encontrarás fragmentos de código y explicaciones para cada uno.

## Índice

1. [Librerias que usaremos en el examen](#librerias-que-usaremos-en-el-examen)
2. [Seleccion de la imagen con CV2](#seleccion-de-la-imagen-con-cv2)
3. [Eliminación de un Canal de Color](#eliminación-de-un-canal-de-color)

## Librerias que usaremos en el examen

Elige una imagen con cv2 y le saca el RGB:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

## Seleccion de la imagen con CV2

Elige una imagen con cv2 y le saca el RGB:

```python
image = cv2.imread('flowers.jpg')
```

## Eliminación de un Canal de Color

```python
image_sin_cielo = image.copy()
image_sin_cielo[:,:,0] = 0
```

