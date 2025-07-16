import cv2
import numpy as np
import os

# Rutas locales (Disco duro externo)
file_H1 = '/media/manuel/Windows/Speckle_H1' # Contiene las 4096 imagenes de los speckles correspondientes a H1
file_H2 = '/media/manuel/Windows/Speckle_H2' # lo mismo para H2


######################### Obtención de sub matriz de intensidad Speckles H1 #########################

# Obtener lista ordenada de archivos PNG
archivos_H1 = sorted([
    f for f in os.listdir(file_H1) if f.endswith('.png')
])

# Dimensiones de imagen
alto, ancho = 1024, 1280
M = alto * ancho
N_H1 = len(archivos_H1)  # número total de imágenes

# Inicializar la matriz de salida (cada columna es un vector speckle)
Submatriz_Intensidad_H1 = np.zeros((M, N_H1), dtype=np.uint8)

# Iterar sobre las imágenes y vectorizarlas
for idx, archivo in enumerate(archivos_H1):
    
    ruta_img = os.path.join(file_H1, archivo)
    
    # Se carga la imagen del speckle en escala de grises
    img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)

    # Verifica que la imagen tenga el tamaño adecuado
    if img.shape != (alto, ancho):
        raise ValueError(f"La imagen {archivo} tiene tamaño inesperado: {img.shape}")

    # Binariza con unmbral 127
    _, Matriz_Binaria_Speckle_H1 = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    # Se convierte la matriz binaria del speckle a un vector columna. (La vectorización se realiza fila a fila porque Jose programo así las matrices de Hadamard.)
    Vector_Binario_Speckle_H1 = Matriz_Binaria_Speckle_H1.flatten(order='C')  # fila a fila
    
    # Se va almacenando cada vector como una columna de esta submatriz H1.
    Submatriz_Intensidad_H1[:, idx] = Vector_Binario_Speckle_H1

# Guardar la matriz como archivo binario NumPy
np.save('/media/manuel/Windows/speckles_H1_vectorizados.npy', Submatriz_Intensidad_H1)

print("Matriz de speckles guardada en: /media/manuel/Windows/speckles_H1_vectorizados.npy")


######################### Obtención de sub matriz de intensidad Speckles H2 #########################

# Obtener lista ordenada de archivos PNG
archivos_H2 = sorted([
    f for f in os.listdir(file_H2) if f.endswith('.png')
])

# Dimensiones de imagen
alto, ancho = 1024, 1280
M = alto * ancho
N_H2 = len(archivos_H2)  # número total de imágenes

# Obtener lista ordenada de archivos PNG
archivos = sorted([
    f for f in os.listdir(file_H2) if f.endswith('.png')
])

# Inicializar la matriz de salida (cada columna es un vector speckle)
Submatriz_Intensidad_H2 = np.zeros((M, N_H2), dtype=np.uint8)

# Iterar sobre las imágenes y vectorizarlas
for idx, archivo in enumerate(archivos):
    
    ruta_img = os.path.join(file_H2, archivo)
    
    # Se carga la imagen del speckle en escala de grises
    img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)

    # Verifica que la imagen tenga el tamaño adecuado
    if img.shape != (alto, ancho):
        raise ValueError(f"La imagen {archivo} tiene tamaño inesperado: {img.shape}")

    # Binariza con unmbral 127
    _, Matriz_Binaria_Speckle_H2 = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    # Se convierte la matriz binaria del speckle a un vector columna. (La vectorización se realiza fila a fila porque Jose programo así las matrices de Hadamard.)
    Vector_Binario_Speckle_H2 = Matriz_Binaria_Speckle_H2.flatten(order='C')  # fila a fila
    
    # Se va almacenando cada vector como una columna de esta submatriz H2.
    Submatriz_Intensidad_H2[:, idx] = Vector_Binario_Speckle_H2

# Guardar la matriz como archivo binario NumPy
np.save('/media/manuel/Windows/speckles_H2_vectorizados.npy', Submatriz_Intensidad_H2)

print("Matriz de speckles guardada en: /media/manuel/Windows/speckles_H2_vectorizados.npy")



