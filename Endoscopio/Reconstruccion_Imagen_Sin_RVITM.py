import numpy as np
import cv2
import os

# Antes de correr el código verificar el path de la imagen a reconstruir, el path y el nombre de salida

# Paths
# Ojo que toca modificar el path de la imagen que toca reconstruir!!!
Path_Imagen_a_Reconstruir = 'D:\\Speckle_De_Imagenes_A_Reconstruir\\pato.png'

Path_Matriz_Intensidad = 'D:\\Archivos_Reconstruccion\\Matriz_Intensidad.npy'
Path_Matriz_Hadamard_T = 'D:\\Archivos_Reconstruccion\\Hadamard_H_menosH_transpuesta.npy'

# Modificar el output path para evitar que se sobreescriban!!!!
Output_Path = 'D:\\Archivos_Reconstruccion\\Imagenes_Reconstruidas'

# Crear carpeta si no existe
os.makedirs(Output_Path, exist_ok=True)

# Cargar matrices a operar
Matriz_Hadamard_T = np.load(Path_Matriz_Hadamard_T)      # (8192, 1310720)
Matriz_Intensidad = np.load(Path_Matriz_Intensidad)        # (1310720, 8192)

# Verificación de dimensiones
assert Matriz_Intensidad.shape[1] == Matriz_Hadamard_T.shape[0], "Dimensiones incompatibles para multiplicación"

# Número de patrones originales de Hadamard
N = Matriz_Hadamard_T.shape[0] // 2

# Leer y binarizar imagen speckle
print(f"Procesando imagen: {os.path.basename(Path_Imagen_a_Reconstruir)}")
img = cv2.imread(Path_Imagen_a_Reconstruir, cv2.IMREAD_GRAYSCALE)
_, binaria = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
I_out = binaria.flatten(order='C').astype(np.float32).reshape(-1, 1)  # (131072, 1)

# Inicializar imagen reconstruida como vector
I_rec = np.zeros((Matriz_Intensidad.shape[0], 1), dtype=np.float32)

# Chunking
chunk_size = 1024   # Variable ajustable en función de la capacidad de la RAM. Si no da bajar a 256
for i in range(0, Matriz_Hadamard_T.shape[1], chunk_size):
    XTi = Matriz_Hadamard_T[:, i:i + chunk_size]    # (8192, chunk)
    I_rec[i:i + chunk_size] = (1 / (2 * N)) * Matriz_Intensidad @ (XTi @ I_out)

# Binarizar reconstrucción
I_bin = (I_rec > 0).astype(np.uint8) * 255
img_rec = I_bin.reshape((1024, 1280))

# Guardar imagen reconstruida
# Cambiar nombre de salida en función de la imagen!!!
nombre_salida = f'reconstruida_{os.path.basename(Path_Imagen_a_Reconstruir)}'
cv2.imwrite(os.path.join(Output_Path, nombre_salida), img_rec)

print(f"Imagen reconstruida guardada como: {nombre_salida}")