import cv2
import numpy as np
import os

# Rutas a los patrones Hadamard proyectados
path_H1 = 'D:\\Hadamard_1_64_1280x1024'
path_H2 = 'D:\\Hadamard_2_64_1280x1024'

# Obtener listas ordenadas de archivos
archivos_H1 = sorted([f for f in os.listdir(path_H1) if f.endswith('.png')])
archivos_H2 = sorted([f for f in os.listdir(path_H2) if f.endswith('.png')])

# Dimensiones conocidas
alto, ancho = 1024, 1280
M = alto * ancho
N = len(archivos_H1)  # deben ser 4096

# Inicializar matrices
H1_bin = np.zeros((M, N), dtype=np.uint8)
H2_bin = np.zeros((M, N), dtype=np.uint8)

# Cargar y vectorizar patrones Hadamard binarios
for idx in range(N):
    
    # Mismo proceso que se realizo en la binarizacion de los Speckles.
    
    # H1
    img_H1 = cv2.imread(os.path.join(path_H1, archivos_H1[idx]), cv2.IMREAD_GRAYSCALE)
    _, bin_H1 = cv2.threshold(img_H1, 127, 1, cv2.THRESH_BINARY)
    H1_bin[:, idx] = bin_H1.flatten(order='C')

    # H2
    img_H2 = cv2.imread(os.path.join(path_H2, archivos_H2[idx]), cv2.IMREAD_GRAYSCALE)
    _, bin_H2 = cv2.threshold(img_H2, 127, 1, cv2.THRESH_BINARY)
    H2_bin[:, idx] = bin_H2.flatten(order='C')

# Construcción matriz [H,-H]

# Convertir de binario a {-1, +1} 
H = 2 * H1_bin - 1  # H ∈ {-1, +1}
neg_H = -(2 * H2_bin - 1)  # -H ∈ {-1, +1}

# Concatenar matriz de entrada: X = [H | -H] y transponer
X_T = np.hstack((H, neg_H)).T  # tamaño (131072, 8192)

# Guardar para reutilizar luego
np.save('D:\\Archivos_Reconstruccion\\Hadamard_H_menosH.npy', X_T)

print("Matriz de entrada Hadamard [H, -H] construida y guardada exitosamente.")