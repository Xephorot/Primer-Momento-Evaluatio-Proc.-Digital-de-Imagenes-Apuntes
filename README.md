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

## Clase 23/02/2023 Deteccion de pixeles
Primer codigo
```python
import numpy as np 
import cv2
import matplotlib.pyplot as plt

#En Escala de grises de una forma extraña

I = cv2.imread("GalacticBackground.png")

img = I[:, :, (2,1,0)]
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

Rd = R.astype(float)
Gd = G.astype(float)
Bd = B.astype(float)

k = (0.1,0.1,0.1)
Zd = k[0]*Rd+k[1]*Gd[2]*Bd

Z = Zd.astype(np.uint8)

#cv2.imshow("HOLA", Z)
#cv2.waitKey()

plt.imshow(Z, cmap="gray")
```
Segundo codigo
```python
import numpy as np 
import cv2
import matplotlib.pyplot as plt

I = cv2.imread("zapato disfuncional.png")

#Con CV2

#cv2.imshow("prueba", I)
#cv2.waitKey()

#Con MatPlot

#Invertir Colores para mostrar la imagen original
img = I[:, :, (2,1,0)]

#Separacion de Colores

#Blanco
R = img[:,:,0]
#Celeste
G = img[:,:,1]
#Verde
B = img[:,:,2]

#Mostrar Imagen original
plt.imshow(img, cmap="gray")

# Imprimir valores máximos y mínimos de los canales de color
print("Valor máximo del canal R:", np.max(R))
print("Valor mínimo del canal R:", np.min(R))
print("Valor máximo del canal G:", np.max(G))
print("Valor mínimo del canal G:", np.min(G))
print("Valor máximo del canal B:", np.max(B))
print("Valor mínimo del canal B:", np.min(B))

print("Valor color del zapato:")
print(img[1400,1000])


'''
# Crear una imagen con el color específico

color = img[1400, 1000]

color_image = np.full((100, 100, 3), color)

# Mostrar la imagen del color específico
plt.imshow(color_image)
'''


```
Tercer codigo
```python
import numpy as np 
import cv2
import matplotlib.pyplot as plt

#En Escala de grises de una forma extraña

I = cv2.imread("GalacticBackground.png")

#Invertir
img = I[:, :, (2,1,0)]

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

#Pintar de otra forma
X = img - 123

plt.imshow(X, cmap="gray")
```

## Clase 26/02/2023 Pixelizacion
```python
import numpy as np 
from cv2 import imread
import matplotlib.pyplot as plt

def detalles(img):
    print("Size = ", img.shape)
    print("Max = ", np.max(img))
    print("Min = ", np.min(img))
    

img = imread("grayscale.png")

X = img [:,:,0]

#plt.figure(figsize=(8,8))

#plt.imshow(X,cmap="gray")
#plt.show()

'''
-----------------------------------------------------------------------------------------
'''

#Tarea
d = 8
(Nx, Mx) = X.shape  # Obtenemos las dimensiones de la matriz X
ix = range(0, Nx, d)  # Creamos rangos para los índices de filas
jx = range(0, Mx, d)  # Creamos rangos para los índices de columnas
'''
Creamos la matriz Y tomando "rebanadas" llamadas slice (partes) de la matriz X.
Cada "rebanada" se forma seleccionando filas en ix y columnas en jx.
'''
Y = np.array([X[x_slice, jx] for x_slice in ix])

'''
-----------------------------------------------------------------------------------------
'''

#Desde aqui es la tarea
'''
d = 8
(Nx, Mx) = X.shape
ix = range(0,Nx,d)
jx = range(0,Mx,d)
Ny = len(ix)
My = len(jx)
Y = np.zeros((Ny,My),np.uint8)
for i in range (Ny):
    for j in range (My):
        Y[i,j] = X[ix[i],jx[j]]
'''
#Hasta aqui


'''

'''

plt.figure(figsize=(8,8))

plt.imshow(Y,cmap="gray")
plt.show()

def histo (X,n = 256):
    (N,M) = X.shape
    h = np.zeros((n,))
    for i in range (N):
        for j in range (M):
            x = X[i,j]
            h[x] = h[x]+1
    return h

n = 256
h = histo(Y, n = n)

plt.figure(figsize=(12,8))
plt.plot(range(n), h[0:n])
plt.show()
```
## Clase 28/02/2024 Seleccion por color
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

def detalles(img):
    print("Size = ", img.shape)
    print("Max = ", np.max(img))
    print("Min = ", np.min(img))
    
def imprimir(imagen):
    plt.imshow(imagen, cmap="gray")
    plt.show()
    
def seleccionar_area_flor(Q, I):
    N, M = Q.shape
    imin, imax, jmin, jmax = 1000, 0, 1000, 0
    for i in range(N):
        for j in range(M):
            if Q[i, j] > 0:
                imin = min(imin, i)
                imax = max(imax, i)
                jmin = min(jmin, j)
                jmax = max(jmax, j)
    y = [imin, imin, imax, imax, imin]
    x = [jmin, jmax, jmax, jmin, jmin]
    plt.imshow(I)
    plt.plot(x, y, color='yellow') # Cambiado para resaltar el contorno
    plt.show()

# Carga y preparación de la imagen
I1 = cv2.imread("flowers.jpg")
I = I1[:, :, (2, 1, 0)] # Ajuste para que coincida con el orden de colores de matplotlib

# Conversión a escala de grises simplificada
Z = np.mean(I.astype(float), axis=2).astype(np.uint8)

# Cambio a amarillo: simplificación de la lógica
Sr, Sg, Sb = I[:, :, 0] > 0, I[:, :, 1] > 150, I[:, :, 2] < 100
S = np.logical_and(np.logical_and(Sr, Sg), Sb)

# Limpiar impurezas
N, M = S.shape
Q = np.copy(S)
for i in range(N):
    if np.sum(S[i, :]) < 30:
        Q[i, :] = 0

# Llamada a la función para seleccionar área de la flor
seleccionar_area_flor(Q, I)
```

## Clase 01/03/2024 Cambio y eliminacion de colores RGB
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Carga y ajuste de la imagen
image = cv2.imread("flowers.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Configuraciones de K-means
k = 3
pixel_values = image.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.3)

# Aplicación de K-means
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Conversión de centros a uint8 y preparación de la imagen segmentada
centers = np.uint8(centers)
labels = labels.flatten()
segmented_image = centers[labels].reshape(image.shape)

# Función para eliminar un color específico
def eliminar_color(segmented_image, centers):
    target_sum = sum(centers[0]) # Suma de componentes del primer centro para identificar el color a eliminar
    for i in range(segmented_image.shape[0]):
        for j in range(segmented_image.shape[1]):
            if sum(segmented_image[i, j]) == target_sum:
                segmented_image[i, j] = [0, 0, 0] # Asignar negro donde la suma de componentes coincide
    return segmented_image

# Eliminación de un color en la imagen segmentada
segmented_color_removed = eliminar_color(segmented_image.copy(), centers)

# Visualización
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Segmented Image")
plt.imshow(segmented_image)

plt.subplot(1, 2, 2)
plt.title("Color Removed")
plt.imshow(segmented_color_removed)

plt.show()

```

## Clase 06/03/2024 Otro tipo de Histrogramas
Codigo 1
```python
import numpy as np 
import matplotlib.pyplot as plt

img = "onerice.png"

imagen = [[255, 0, 255],
          [255, 0, 255],
          [255, 0, 255]]

imagen_array = np.array(imagen)

# Imagen
plt.imshow(imagen_array, cmap='gray')
plt.show()

# Histograma "bonito" 
def histo(X, n=256):
    (N, M) = X.shape
    h = np.zeros((n,))
    for i in range(N):
        for j in range(M):
            x = X[i, j]
            h[x] = h[x] + 1
    plt.plot(range(n), h[0:n])
    plt.show()

histo(img)

#Histograma en barras
plt.hist(imagen_array, range=(0,256))
plt.show()
```
Codigo 2
```python
import numpy as np
from cv2 import imread
import matplotlib.pyplot as plt
 
img = imread("onerice.bmp")
plt.imshow(img)
 
def detalles (img):
    print("tam= ",img.shape)
    print("max: ", np.max(img))
    print("min: ",np.min(img))
X = img[:,:,0]
 
def segmenta (x,t):
    (F,C)=x.shape
    Y = np.zeros((F,C),np.uint8)
    area = 0
    for i in range (F):
        for j in range (C):
            if x[i,j] > t:
                Y[i,j]= 1
                area = area+1
    print("area: ",area)
    return Y

#Histograma Bonito
def histo (X):
    (N,M) = X.shape
    n=257
    h = np.zeros((n,),np.uint8)
    for i in range (N):
        for j in range (M):
            x = X[i,j]
            h[x]= h[x] + 1
    plt.plot(range(n),h[0:n])       
    plt.show()        
Y = segmenta(X,120)
 
#plt.colorbar()
plt.imshow (X,cmap="gray")  
plt.show()
histo(X)
plt.show()

#Histograma otra manera
L = X
#Sacar el tamaño
print(L.shape) # 64
#Convertir N, M a L
(N,M) = L.shape

#Verificar N y M
print("N: ", N) # 64
print("M: ", M) # 64

#Se multiplica N y M para re ordenar y que entren los 9 elementos en un array
L.shape = (N * M)

plt.hist(L, bins = 255, range=(0,255),histtype="step")
```
Codigo 3
```python
import numpy as np
from cv2 import imread
import matplotlib.pyplot as plt
 
img = imread("flowers.jpg")

img_arreglada = img [:,:,(2,1,0)]

img_capas = img[:,:,0]

#Histograma "Bonito"
def histo (X):
    (N,M) = X.shape
    n=257
    h = np.zeros((n,),np.uint8)
    for i in range (N):
        for j in range (M):
            x = X[i,j]
            h[x]= h[x] + 1
    plt.plot(range(n),h[0:n])       
    plt.show()   
    
def detalles(img):
    print("Size = ", img.shape)
    print("Max = ", np.max(img))
    print("Min = ", np.min(img))
    
def Imprimir(imagen):
    plt.imshow(imagen,cmap = "gray")
    plt.show()
    
plt.imshow (img_arreglada,cmap="gray")  
plt.show()
histo(img_capas)
plt.show()

#RGB
#Capa 0
R = img[:,:,0]
#Capa 1
G = img[:,:,1]
#Capa 2
B = img[:,:,2]

#Histograma otra manera
plt.hist(R.flatten(), bins=255, range=(10, 250), histtype="step", color = "red")
plt.hist(G.flatten(), bins=255, range=(10, 250), histtype="step", color = "green")
plt.hist(B.flatten(), bins=255, range=(10, 250), histtype="step", color = "blue")
plt.show()
```

## Primer Examen
Ejercicio 1
```python
import numpy as np
from cv2 import imread
import matplotlib.pyplot as plt
 
#Histograma "Bonito"
def histo (X):
    (N,M) = X.shape
    n=257
    h = np.zeros((n,),np.uint8)
    for i in range (N):
        for j in range (M):
            x = X[i,j]
            h[x]= h[x] + 1
    plt.plot(range(n),h[0:n])       
    plt.show()   
    
def detalles(img):
    print("Size = ", img.shape)
    print("Max = ", np.max(img))
    print("Min = ", np.min(img))
    
def Imprimir(imagen):
    plt.imshow(imagen,cmap = "gray")
    plt.show()

#Frutos Azules
img = imread("FRUTA AZUL.JPG")

img_arreglada = img [:,:,(2,1,0)]

img_capas = img[:,:,0]

#Capa 0
R = img_arreglada[:,:,0]
#Capa 1
G = img_arreglada[:,:,1]
#Capa 2
B = img_arreglada[:,:,2]

#Centrar al color azul
Sr = R > 40
Sg = G > 0
Sb = B < 100

Srgb = np.concatenate((Sr,Sg,Sb), axis = 1)

#Limpiando lo amarillo con lógica (AND)
Srg = np.logical_and(Sr,Sg)
S = np.logical_and(Srg, Sb)

(N,M) = S.shape

#Limpiando Impurezas por fila (Pixeles faltantes)
Q = S

for i in range(N):
    s = np.sum(S[i,:])
    if s < 30:
        Q[i,:] = 0

imin = 1000
imax = 0
jmin = 1000
jmax = 0

(N,M) = S.shape

#Seleccion
for i in range (N):
    for j in range(M):
        if Q[i,j] > 0:
            if i < imin:
                imin = i
            if i > imax:
                imax = i
            if j < jmin:
                jmin = j
            if j > jmax:
                jmax = j

y = [imin, imin, imax, imax, imin]

x = [jmin, imax, imax, jmin, jmin]

#Pintar borde de la imagen
E = np.zeros((N,M),np.uint8)

for i in range(N):
    for j in range(1,M):
        if Q[i,j] != Q[i,j-1]:
            E[i,j]=1
            E[i,j-1]=1
        

for i in range(1,N):
    for j in range(M):
        if Q[i,j] != Q[i-1,j]:
            E[i,j]=1
            E[i-1,j]=1


plt.imshow(E, cmap="gray")

for i in range (N):
    for j in range (M):
        if E[i,j] == 1:
            img_arreglada[i,j,:] = (0,0,1)


#Imprimir en Escala de grises
Rd = R.astype(float)
Gd = G.astype(float)
Bd = B.astype(float)

Zd = 1/3 * Rd + 1/3 * Gd + 1/3 * Bd

Z = Zd.astype(np.uint8)

plt.imshow (Z,cmap="gray")  
plt.show()
            

#Histograma
plt.hist(R.flatten(), bins=255, range=(10, 250), histtype="step", color = "red")
plt.hist(G.flatten(), bins=255, range=(10, 250), histtype="step", color = "green")
plt.hist(B.flatten(), bins=255, range=(10, 250), histtype="step", color = "blue")
plt.show()
```
Ejercicio 2
```python
import numpy as np
from cv2 import imread
import matplotlib.pyplot as plt
 
#Histograma "Bonito"
def histo (X):
    (N,M) = X.shape
    n=257
    h = np.zeros((n,),np.uint8)
    for i in range (N):
        for j in range (M):
            x = X[i,j]
            h[x]= h[x] + 1
    plt.plot(range(n),h[0:n])       
    plt.show()   
    
def detalles(img):
    print("Size = ", img.shape)
    print("Max = ", np.max(img))
    print("Min = ", np.min(img))
    
def Imprimir(imagen):
    plt.imshow(imagen,cmap = "gray")
    plt.show()

#Gato
img = imread("gato negro.jpg")

img_arreglada = img [:,:,(2,1,0)]

img_capas = img[:,:,0]

#Capa 0
R = img_arreglada[:,:,0]
#Capa 1
G = img_arreglada[:,:,1]
#Capa 2
B = img_arreglada[:,:,2]

#Imprimir(R)

#Seleccion al gato (Negro)

Sr = R < 75
Sg = G < 80
Sb = B < 81

Srgb = np.concatenate((Sr,Sg,Sb), axis = 1)

#Limpiando lo amarillo con lógica (AND)
Srg = np.logical_and(Sr,Sg)
S = np.logical_and(Srg, Sb)

(N,M) = S.shape

#Limpiando Impurezas por fila (Pixeles faltantes)
Q = S 

for i in range(N):
    s = np.sum(S[i,:])
    if s < 0:
        Q[i,:] = 0
        
#Imprimir(Q)

imin = 1000
imax = 0
jmin = 1000
jmax = 0

#Tarea

(N,M) = S.shape

for i in range (N):
    for j in range(M):
        if Q[i,j] > 0:
            if i < imin:
                imin = i
            if i > imax: 
                imax = i + 345
            if j < jmin:
                jmin = j
            if j > jmax:
                jmax = j

#No especifico no modificar esta parte inge, dijo solo enmarcar y en escala de grises.
y = [imin, imin, imax - 210, imax - 210, imin]

x = [jmin, imax, imax, jmin, jmin]

#Imprimir en Escala de grises
Rd = R.astype(float)
Gd = G.astype(float)
Bd = B.astype(float)

Zd = 1/3 * Rd + 1/3 * Gd + 1/3 * Bd

Z = Zd.astype(np.uint8)

plt.imshow (Z,cmap="gray")  
plt.plot(x,y)
plt.show()

#Histograma
plt.hist(R.flatten(), bins=255, range=(10, 250), histtype="step", color = "red")
plt.hist(G.flatten(), bins=255, range=(10, 250), histtype="step", color = "green")
plt.hist(B.flatten(), bins=255, range=(10, 250), histtype="step", color = "blue")
plt.show()
```
## 11/03/2024 Ordenacion de imagenes con Linear y reordenacion
```python
import numpy as np
import cv2

# Cargar la imagen
image = cv2.imread("gato negro.jpg")

# Factores de escalado
ep = 0.6  # Para reducir
eg = 1.6  # Para aumentar

# Redimensionar imagen: reducción y ampliación
img_p = cv2.resize(image, None, fx=ep, fy=ep, interpolation=cv2.INTER_LINEAR)
img_g = cv2.resize(image, None, fx=eg, fy=eg, interpolation=cv2.INTER_LINEAR)

# Ampliación con diferentes interpolaciones
img_g2 = cv2.resize(image, None, fx=eg, fy=eg, interpolation=cv2.INTER_NEAREST)
img_g3 = cv2.resize(image, None, fx=eg, fy=eg, interpolation=cv2.INTER_AREA)

# Sección de la imagen ampliada para detalle
l = img_g[100:300, 150:450]

# Partir la imagen original en 9 partes y reordenarlas aleatoriamente
parts = [cv2.resize(image[i * image.shape[0] // 3:(i + 1) * image.shape[0] // 3,
                           j * image.shape[1] // 3:(j + 1) * image.shape[1] // 3],
                    (100, 100), interpolation=cv2.INTER_LINEAR) 
         for i in range(3) for j in range(3)]
np.random.shuffle(parts)

# Reconstruir la imagen a partir de las partes reordenadas
combined_image = np.vstack([np.hstack(parts[i:i+3]) for i in range(0, 9, 3)])

# Mostrar las imágenes resultantes
cv2.imshow("Reducida", img_p)
cv2.imshow("Ampliada con INTER_LINEAR", img_g)
cv2.imshow("Detalle de Ampliada", l)
cv2.imshow("Ampliada con INTER_NEAREST", img_g2)
cv2.imshow("Ampliada con INTER_AREA", img_g3)
cv2.imshow("Imagen Reconstruida", combined_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
## 13/03/2024 Utilizacion de diferente histograma junto Aumento de brillo de imagen
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen
img = cv2.imread("Cartoon Bright Kid.jpg");

# Seleccionar solo la capa B de la imagen (en OpenCV es BGR, así que B es el índice 0)
img1 = img[:, :, 0]

# Visualización del histograma original
plt.figure(figsize=(10, 3))
plt.hist(img1.ravel(), bins=256, color='blue', alpha=0.5, label='Original')
plt.legend()
plt.title("Histograma Original")
plt.show()

# Método 1: Estiramiento del histograma
X1 = 255 * (img1 - img1.min()) / (img1.max() - img1.min())
X1 = X1.astype(np.uint8)

# Método 2: Ecualización del histograma usando OpenCV
X3 = cv2.equalizeHist(img1)

# Visualización de las imágenes lado a lado
res = np.hstack((img1, X1, X3))
plt.figure(figsize=(10, 3))
plt.imshow(res, cmap='gray')
plt.title("Original - Estirada - Ecualizada")
plt.show()

# Visualización de los histogramas
# Histograma del niño claro
plt.figure(figsize=(10, 3))
plt.hist(X1.ravel(), bins=256, color='green', alpha=0.5, label='Estirado')
# Histograma del niño oscuro (Método 1 y Método 2)
plt.hist(X3.ravel(), bins=256, color='red', alpha=0.5, label='Ecualizado')
plt.legend()
plt.title("Histogramas Estirado y Ecualizado")
plt.show()
```
