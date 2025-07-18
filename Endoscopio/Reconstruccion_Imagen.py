import cv2
import numpy as np
import os

# Rutas
path_speckles = '/media/manuel/Windows/Speckle_De_Imagenes_A_Reconstruir'
ruta_rvitm = '/media/manuel/Windows/RVITM_estimada.npy'
ruta_salida = '/media/manuel/Windows/Imagenes_Reconstruidas'

# Crear carpeta de salida si no existe
os.makedirs(ruta_salida, exist_ok=True)

# Cargar la RVITM en float32
print("Cargando matriz RVITM...")
RVITM = np.load(ruta_rvitm).astype(np.float32)  # (1310720, 1310720)

# Listar imágenes de speckles
archivos = sorted([f for f in os.listdir(path_speckles) if f.endswith('.png')])

# Procesar cada imagen de speckle
for nombre in archivos:
    print(f"Procesando {nombre}...")

    # Leer y binarizar la imagen de speckle
    ruta_img = os.path.join(path_speckles, nombre)
    img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
    _, binaria = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    # Vectorizar y convertir a float32
    I_out = binaria.flatten(order='C').astype(np.float32).reshape(-1, 1)  # (131072, 1)

    # Reconstruir patrón de entrada
    I_in_est = (RVITM.T @ I_out) / 8192  # resultado en float32

    # Umbral para binarizar reconstrucción
    I_in_bin = (I_in_est > 0).astype(np.uint8) * 255  # volver a imagen 8 bits

    # Convertir de vector a imagen (reshape)
    img_rec = I_in_bin.reshape((1024, 1280))  # (alto, ancho)

    # Guardar reconstrucción
    nombre_salida = f'reconstruida_{nombre}'
    cv2.imwrite(os.path.join(ruta_salida, nombre_salida), img_rec)

    print(f"Imagen reconstruida guardada: {nombre_salida}")
