import cv2
import numpy as np
import os

# Rutas a los patrones Hadamard proyectados
path_H1 = '/media/manuel/Windows/Hadamard_1_64_1280x1024'
path_H2 = '/media/manuel/Windows/Hadamard_2_64_1280x1024'

# Obtener listas ordenadas de archivos
archivos_H1 = sorted([f for f in os.listdir(path_H1) if f.endswith('.png')])
archivos_H2 = sorted([f for f in os.listdir(path_H2) if f.endswith('.png')])

# Dimensiones conocidas
alto, ancho = 1024, 1280
M = alto * ancho
N1 = len(archivos_H1)  # deben ser 4096
N2 = len(archivos_H2)  # deben ser 4096

print("Numero de archivos en archivos_H1:")
print(N1)


print("Numero de archivos en archivos_H2:")
print(N2)

if(N1 != N2):
    print("Tamaño de matrices H1 & H2 diferente")


# Inicializar matrices
H1_bin = np.zeros((M, N1), dtype=np.uint8)
H2_bin = np.zeros((M, N2), dtype=np.uint8)


# Cargar y vectorizar patrones Hadamard binarios
for idx in range(N1):
    
    # Mismo proceso que se realizo en la binarizacion de los Speckles.
    
    # H1
    img_H1 = cv2.imread(os.path.join(path_H1, archivos_H1[idx]), cv2.IMREAD_GRAYSCALE)
    _, bin_H1 = cv2.threshold(img_H1, 127, 1, cv2.THRESH_BINARY)
    H1_bin[:, idx] = bin_H1.flatten(order='C')

    # H2
    img_H2 = cv2.imread(os.path.join(path_H2, archivos_H2[idx]), cv2.IMREAD_GRAYSCALE)
    _, bin_H2 = cv2.threshold(img_H2, 127, 1, cv2.THRESH_BINARY)
    H2_bin[:, idx] = bin_H2.flatten(order='C')

# Construcción matriz [H, -H] y su transpuesta

# Convertir de binario a {-1, +1}
H = 2 * H1_bin - 1  # H ∈ {-1, +1}
neg_H = -(2 * H2_bin - 1)  # -H ∈ {-1, +1}

# Concatenar y transponer directamente
X_T = np.hstack((H, neg_H)).T.astype(np.float32)  # Xᵀ ∈ (8192, 131072)

# Guardar la transpuesta directamente
np.save('/media/manuel/Windows/Hadamard_H_menosH_transpuesta.npy', X_T)

print("Matriz transpuesta Hadamard [H, -H]ᵀ guardada exitosamente.")
