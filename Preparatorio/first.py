import numpy as np
import cv2
import matplotlib.pyplot as plt

def imprimir(imagen):
    plt.imshow(imagen, cmap="gray")
    plt.show()

def seleccionar_area_objeto(Q, I):
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

image = cv2.imread('Preparatorio/image.png')
I = image[:, :, (2, 1, 0)] 

Z = np.mean(I.astype(float), axis=2).astype(np.uint8)

Sr, Sg, Sb = I[:, :, 0] > 100, I[:, :, 1] < 150, I[:, :, 2] < 100
S = np.logical_and(np.logical_and(Sr, Sg), Sb)

N, M = S.shape
Q = np.copy(S)
for i in range(N):
    if np.sum(S[i, :]) < 30:
        Q[i, :] = 0

imprimir(Sb)
seleccionar_area_objeto(Q, I)
