import cv2
import numpy as np
import os

'''
Este código realiza la vectorización de los patrones speckle capturados.
Recibe imágenes PNG de tamaño 1280 x 1024 y genera matrices de tamaño 1310720 x 4096. 
Estas no se binarizan, sino que se guardan manteniendo los valores de intensidad originales
(0-255) en formato uint8.
'''
# Rutas locales (Disco duro externo)
file_H1 = 'D:\\Speckle_H1_1280x1024' # Contiene las 4096 imagenes de los speckles respuesta a H1
file_H2 = 'D:\\Speckle_H2_1280x1024' # Contiene las 4096 imagenes de los speckles respuesta a H2

######################### Obtención de sub matriz de intensidad Speckles H1 #########################

# Obtener lista ordenada de archivos PNG
archivos_H1 = sorted([
    f for f in os.listdir(file_H1) if f.endswith('.png')
])

# Dimensiones de imagen (alto, ancho) según formato OpenCV
alto, ancho = 1024, 1280  # OpenCV usa (filas, columnas) = (alto, ancho)
M = alto * ancho    # 1310720
N_H1 = len(archivos_H1)  # número total de imágenes (deben ser 4096)

'''
Validaciones para prevenir overflows y asegurar coherencia dimensional
'''
print(f"=== PROCESANDO SPECKLES H1 ===")
print(f"Dimensiones detectadas: {alto}x{ancho}, M={M}, N_H1={N_H1}")

# Calcular memoria requerida para speckles H1
# Cálculo: 1 matriz de M×N elementos, cada elemento uint8 (1 byte)
# Fórmula: M * N / (1024³) donde 1024³ = bytes a GB
memoria_H1_gb = M * N_H1 / (1024**3)
print(f"Memoria requerida para Submatriz_Intensidad_H1: {memoria_H1_gb:.2f} GB")

# Verificar que las dimensiones son coherentes
if M > 2**31 - 1:  # Límite de int32
    raise ValueError(f"M={M} excede el límite de int32")
if N_H1 > 2**15:  # Límite práctico para evitar matrices muy grandes
    print(f"Advertencia: N_H1={N_H1} es muy grande.")

'''
Genera una matriz de intensidades de tamaño M x N_H1 de variables de tipo uint8.
Donde cada columna es un vector speckle con valores de intensidad (0-255).
Se inicializa con ceros.
'''
# Inicializar la matriz de salida (cada columna es un vector speckle)
Submatriz_Intensidad_H1 = np.zeros((M, N_H1), dtype=np.uint8)

'''
Se hace entonces una iteración para procesar las 4096 imágenes correspondientes a la respuesta (speckle) 
de la MMF al patrón de Hadamard.
Cada imagen se carga manteniendo sus valores de intensidad originales y se almacena como una columna vectorizada.
'''
# Iterar sobre las imágenes y vectorizarlas
for idx, archivo in enumerate(archivos_H1):
    
    # Se busca la imagen en la ruta
    ruta_img = os.path.join(file_H1, archivo)
    
    # Se carga la imagen del speckle en escala de grises
    img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)

    # Validaciones de carga e integridad de imagen
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {archivo}")
    if img.shape != (alto, ancho):
        raise ValueError(f"Imagen {archivo} tiene dimensiones {img.shape}, se esperaba ({alto}, {ancho})")

    # Verificar que los valores están en el rango esperado para uint8
    if img.min() < 0 or img.max() > 255:
        raise ValueError(f"Imagen {archivo} tiene valores fuera del rango uint8: [{img.min()}, {img.max()}]")

    # Se convierte la matriz del speckle a un vector columna
    # La vectorización se realiza fila a fila (order='C')
    Vector_Intensidad_Speckle_H1 = img.flatten(order='C')  # fila a fila
    
    # Se va almacenando cada vector como una columna de esta submatriz H1
    Submatriz_Intensidad_H1[:, idx] = Vector_Intensidad_Speckle_H1
    
    # Mostrar progreso cada 100 imágenes
    if (idx + 1) % 100 == 0:
        print(f"H1: Procesadas {idx + 1}/{N_H1} imágenes")

# Validar matriz final H1
print(f"Validando matriz H1 final:")
print(f"Forma: {Submatriz_Intensidad_H1.shape}")
print(f"Rango de valores: [{Submatriz_Intensidad_H1.min()}, {Submatriz_Intensidad_H1.max()}]")
print(f"Tipo de dato: {Submatriz_Intensidad_H1.dtype}")

# Guardar la matriz como archivo binario NumPy
np.save('D:\\Archivos_Reconstruccion\\speckles_H1_vectorizados.npy', Submatriz_Intensidad_H1)

print("Matriz H1 de speckles guardada exitosamente en: D:\\Archivos_Reconstruccion\\speckles_H1_vectorizados.npy")


######################### Obtención de sub matriz de intensidad Speckles H2 #########################

# Obtener lista ordenada de archivos PNG
archivos_H2 = sorted([
    f for f in os.listdir(file_H2) if f.endswith('.png')
])

# Dimensiones de imagen (reutilizando las mismas variables)
N_H2 = len(archivos_H2)  # número total de imágenes (deben ser 4096)

'''
Validaciones para prevenir overflows H2
'''
print(f"\n=== PROCESANDO SPECKLES H2 ===")
print(f"Dimensiones detectadas: {alto}x{ancho}, M={M}, N_H2={N_H2}")

# Calcular memoria requerida para H2
memoria_H2_gb = M * N_H2 / (1024**3)
print(f"Memoria requerida para Submatriz_Intensidad_H2: {memoria_H2_gb:.2f} GB")

# Verificar coherencia entre H1 y H2
if N_H1 != N_H2:
    print(f"Advertencia: N_H1={N_H1} y N_H2={N_H2} son diferentes")

if N_H2 > 2**15:  # Límite práctico para evitar matrices muy grandes
    print(f"Advertencia: N_H2={N_H2} es muy grande.")

# Inicializar la matriz de salida (cada columna es un vector speckle)
Submatriz_Intensidad_H2 = np.zeros((M, N_H2), dtype=np.uint8)

# Iterar sobre las imágenes y vectorizarlas
for idx, archivo in enumerate(archivos_H2):
    
    # Se busca la imagen en la ruta
    ruta_img = os.path.join(file_H2, archivo)
    
    # Se carga la imagen del speckle en escala de grises
    img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)

    # Validaciones de carga e integridad de imagen
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {archivo}")
    if img.shape != (alto, ancho):
        raise ValueError(f"Imagen {archivo} tiene dimensiones {img.shape}, se esperaba ({alto}, {ancho})")

    # Verificar que los valores están en el rango esperado para uint8
    if img.min() < 0 or img.max() > 255:
        raise ValueError(f"Imagen {archivo} tiene valores fuera del rango uint8: [{img.min()}, {img.max()}]")

    # Se convierte la matriz del speckle a un vector columna
    # La vectorización se realiza fila a fila (order='C')
    Vector_Intensidad_Speckle_H2 = img.flatten(order='C')  # fila a fila
    
    # Se va almacenando cada vector como una columna de esta submatriz H2
    Submatriz_Intensidad_H2[:, idx] = Vector_Intensidad_Speckle_H2
    
    # Mostrar progreso cada 100 imágenes
    if (idx + 1) % 100 == 0:
        print(f"H2: Procesadas {idx + 1}/{N_H2} imágenes")

# Validar matriz final H2
print(f"Validando matriz H2 final:")
print(f"Forma: {Submatriz_Intensidad_H2.shape}")
print(f"Rango de valores: [{Submatriz_Intensidad_H2.min()}, {Submatriz_Intensidad_H2.max()}]")
print(f"Tipo de dato: {Submatriz_Intensidad_H2.dtype}")

# Calcular memoria total utilizada
memoria_total_gb = memoria_H1_gb + memoria_H2_gb
print(f"\n=== RESUMEN FINAL ===")
print(f"Memoria total utilizada: {memoria_total_gb:.2f} GB")
print(f"Matrices H1 y H2 procesadas exitosamente")

# Guardar la matriz como archivo binario NumPy
np.save('D:\\Archivos_Reconstruccion\\speckles_H2_vectorizados.npy', Submatriz_Intensidad_H2)

print("Matriz H2 de speckles guardada exitosamente en: D:\\Archivos_Reconstruccion\\speckles_H2_vectorizados.npy")



