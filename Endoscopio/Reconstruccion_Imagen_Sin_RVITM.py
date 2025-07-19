import numpy as np
import cv2
import os

# === Paths ===
Path_Imagen_a_Reconstruir = 'D:\\Speckle_De_Imagenes_A_Reconstruir\\pato.png'
Path_Matriz_Intensidad = 'D:\\Archivos_Reconstruccion\\Matriz_Intensidad.npy'
Path_Matriz_Hadamard_T = 'D:\\Archivos_Reconstruccion\\Hadamard_H_menosH_transpuesta.npy'
Output_Path = 'D:\\Archivos_Reconstruccion\\Imagenes_Reconstruidas'
os.makedirs(Output_Path, exist_ok=True)

# === Dimensiones conocidas ===
shape_I = (1310720, 8192)
shape_H = (8192, 1310720)

# === Memmap ===
Matriz_Intensidad = np.lib.format.open_memmap(Path_Matriz_Intensidad, dtype='float32', mode='r', shape=shape_I)
Matriz_Hadamard_T = np.lib.format.open_memmap(Path_Matriz_Hadamard_T, dtype='float32', mode='r', shape=shape_H)

# === Imagen a reconstruir ===
print(f"Procesando imagen: {os.path.basename(Path_Imagen_a_Reconstruir)}")
img = cv2.imread(Path_Imagen_a_Reconstruir, cv2.IMREAD_GRAYSCALE)
I_out = img.flatten(order='C').astype(np.float32).reshape(-1, 1)  # Cambiar a float32

# === Cálculo común ===
N = Matriz_Hadamard_T.shape[0] // 2
intermedia = (Matriz_Hadamard_T @ I_out).astype(np.float32)           # float32

# === Reconstrucción con chunks por filas de Matriz_Intensidad ===
I_rec = np.zeros((shape_I[0], 1), dtype=np.float32)  # Resultado final en float32
chunk_size = 65536  # Ajustar según RAM disponible

for i in range(0, shape_I[0], chunk_size):
    fila = Matriz_Intensidad[i:i + chunk_size, :].astype(np.float32)  # float32 solo en esta porción
    I_rec[i:i + chunk_size] = ((1 / (2 * N)) * fila @ intermedia).astype(np.float32)  # Resultado final en float16
    

# === Binarizar y guardar ===
I_bin = (I_rec > 0).astype(np.uint8) * 255
img_rec = I_bin.reshape((1024, 1280))

nombre_salida = f'reconstruida_{os.path.basename(Path_Imagen_a_Reconstruir)}'
cv2.imwrite(os.path.join(Output_Path, nombre_salida), img_rec)
print(f"Imagen reconstruida guardada como: {nombre_salida}")
