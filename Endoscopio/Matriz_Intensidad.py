import cv2
import numpy as np
import os

######################### Construcción de matriz de intensidad #########################

print("Inicia construccion de la matriz de intensidad")

H1_speckles = np.load('speckles_H1_vectorizados.npy')  # (131072, 4096)
H2_speckles = np.load('speckles_H2_vectorizados.npy')  # (131072, 4096)

# Vector columna base correspondiente al speckle de la matriz de Hadamard con todos los pixeles en 1
I1 = H1_speckles[:, 0].reshape(-1, 1)  

# Aplicar fórmula: 2 * I^p - I^1
Y_H1 = 2 * H1_speckles - I1
Y_H2 = 2 * H2_speckles - I1

# Concatenar horizontalmente
Matriz_Intensidad = np.hstack((Y_H1, Y_H2))  # matriz de intensidad 

print("Concatenacion terminada :D")

# Guardar la matriz de intensidad Y
np.save('/media/manuel/Windows/Y_intensidad.npy', Matriz_Intensidad)

print("Matriz de intensidad Y guardada exitosamente.")



