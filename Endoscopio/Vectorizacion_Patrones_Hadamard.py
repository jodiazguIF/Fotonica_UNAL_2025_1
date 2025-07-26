import cv2
import numpy as np
import os
import graficas as graph

'''
Este código realiza la vectorización de los patrones de Hadamard proyectados en el DMD.
Recibe entonces pngs de tamaño 1280 x 1024 y genera dos matrices de tamaño 1310720 x 4096
que luego concatena.
'''

# Rutas a los patrones Hadamard proyectados11
path_H1 = 'D:\\Hadamard_1_64_1280x1024'
path_H2 = 'D:\\Hadamard_2_64_1280x1024'

# Obtener listas ordenadas de archivos
archivos_H1 = sorted([f for f in os.listdir(path_H1) if f.endswith('.png')])
archivos_H2 = sorted([f for f in os.listdir(path_H2) if f.endswith('.png')])

# Dimensiones conocidas
alto, ancho = 1024, 1280
M = alto * ancho    # 1310720
N = len(archivos_H1)  # deben ser 4096

'''
Para poder debuggear el código se integran funciones de validación en las que se imprimen 
dimensiones y memoria requerida.
El objetivo es evitar problemas de overflow y asegurar que las dimensiones son coherentes con 
lo esperado.
'''
# Validaciones para prevenir overflows
print(f"Dimensiones detectadas: {alto}x{ancho}, M={M}, N={N}")

# Calcular memoria requerida
# Cálculo: 2 matrices (H1_bin, H2_bin) de M×N elementos, cada elemento int8 (1 byte)
# Fórmula: 2 * M * N / (1024³) donde 1024³ = bytes a GB
memoria_requerida_gb = 2 * M * N / (1024**3)
print(f"Memoria requerida por matriz individual: {M * N / (1024**3):.2f} GB")
print(f"Memoria total para las matrices principales: {memoria_requerida_gb:.2f} GB")

# Verificar que las dimensiones son razonables
if M > 2**31 - 1:  # Límite de int32
    raise ValueError(f"M={M} excede el límite de int32")
if N > 2**15:  # Límite práctico para evitar matrices muy grandes
    print(f"Advertencia: N={N} es muy grande.")

'''
Genera una matriz binaria de tamaño M x N de variables de tipo int8.
Donde se espera que cada columna sea un vector binario de un patrón Hadamard.
Se inicializa con ceros.
'''
H1_bin = np.zeros((M, N), dtype=np.int8)
H2_bin = np.zeros((M, N), dtype=np.int8)


'''
Se hace entonces una iteración para procesar las 4096*2 imágenes de los patrones de Hadamard.
Cada imagen se carga, se binariza y se almacena como una columna en las matrices H1_bin y H2_bin.
'''
for idx in range(N):
 
    ### H1 ###
    # Se carga la imagen del enésimo patrón Hadamard H1. Lo lee en escala de grises.
    img_H1 = cv2.imread(os.path.join(path_H1, archivos_H1[idx]), cv2.IMREAD_GRAYSCALE)
    
    # Se añade una validación para asegurar que la imagen se cargó correctamente y tiene el tamaño esperado.
    if img_H1 is None:
        raise ValueError(f"No se pudo cargar la imagen: {archivos_H1[idx]}")
    if img_H1.shape != (alto, ancho):
        raise ValueError(f"Imagen {archivos_H1[idx]} tiene dimensiones {img_H1.shape}, se esperaba ({alto}, {ancho})")
    
    # Se binariza la imagen, convirtiendo los valores a 0 o 1 utilizando un umbral de 127.
    bin_H1 = (img_H1 > 127).astype(np.int8)
    
    '''
    Se convierte la matriz binaria del patrón Hadamard a un vector columna.
    La vectorización se realiza fila a fila porque Jose programo así las matrices de Hadamard.
    Se va almacenando cada vector como una columna de esta submatriz H1
    '''
    H1_bin[:, idx] = bin_H1.flatten(order='C')
    
    ### H2 ###
    # Se carga la imagen del enésimo patrón Hadamard H2. Lo lee en escala de grises.
    img_H2 = cv2.imread(os.path.join(path_H2, archivos_H2[idx]), cv2.IMREAD_GRAYSCALE)
    
    # Se añade una validación para asegurar que la imagen se cargó correctamente y tiene el tamaño esperado.
    if img_H2 is None:
        raise ValueError(f"No se pudo cargar la imagen: {archivos_H2[idx]}")
    if img_H2.shape != (alto, ancho):
        raise ValueError(f"Imagen {archivos_H2[idx]} tiene dimensiones {img_H2.shape}, se esperaba ({alto}, {ancho})")
    
    # Same as H1
    bin_H2 = (img_H2 > 127).astype(np.int8)
    H2_bin[:, idx] = bin_H2.flatten(order='C')
    
    # Mostrar progreso cada 100 imágenes
    if (idx + 1) % 100 == 0:
        print(f"Procesadas {idx + 1}/{N} imágenes")

'''
El objetivo es construir una matriz [H,-H], 
Donde H = 2 * H_1 -1    &      -H = 2 * H_2 -1
'''
H = (2 * H1_bin.astype(np.int8)) - 1         # H ∈ {-1, +1}
neg_H = (2 * H2_bin.astype(np.int8)) - 1     # -H ∈ {-1, +1}

# Validación adicional antes de concatenar
print(f"Se validan los rangos de H y -H:")
print(f"H: min={H.min()}, max={H.max()}")
print(f"neg_H: min={neg_H.min()}, max={neg_H.max()}")

# Verificar que los valores están en el rango esperado para int8
if H.min() < -128 or H.max() > 127:
    raise ValueError(f"H tiene valores fuera del rango int8: [{H.min()}, {H.max()}]")
if neg_H.min() < -128 or neg_H.max() > 127:
    raise ValueError(f"neg_H tiene valores fuera del rango int8: [{neg_H.min()}, {neg_H.max()}]")

# Concatenar matriz de entrada: X = [H | -H] y transponer
print(f"Creando matriz concatenada de tamaño: {H.shape[1] + neg_H.shape[1]} x {H.shape[0]}")
print(f"Memoria estimada: {(H.shape[1] + neg_H.shape[1]) * H.shape[0] * 1 / (1024**3):.2f} GB")

try:
    # Validar que el tamaño de la matriz resultante no cause overflow
    filas_finales = H.shape[1] + neg_H.shape[1]  # 8192
    columnas_finales = H.shape[0]                # 1310720
    
    # Verificar overflow en índices de arrays
    if filas_finales * columnas_finales > np.iinfo(np.intp).max:
        raise ValueError(f"El tamaño de la matriz ({filas_finales} x {columnas_finales}) "
                        f"excede el límite máximo de elementos en un array numpy")
    
    X_T = np.hstack((H, neg_H)).T.astype(np.int8)  # tamaño real: (8192, 1310720)
    print(f"Matriz creada exitosamente con forma: {X_T.shape}")
except MemoryError:
    print("Error: No hay suficiente memoria para crear la matriz completa.")
    raise

# Se guarda
np.save('D:\\Archivos_Reconstruccion\\Hadamard_H_menosH_transpuesta.npy', X_T)

print("Matriz de entrada Hadamard [H, -H] construida y guardada exitosamente :D")