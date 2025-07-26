import cv2
import numpy as np
import os
import time
import graficas as graph

'''
Este código realiza la vectorización de los patrones de Hadamard proyectados en el DMD.
Recibe entonces pngs de tamaño 1280 x 1024 y genera dos matrices de tamaño 1310720 x 4096
que luego concatena.
'''

# Rutas a los patrones Hadamard proyectados (Volumen Windows montado en Linux)
path_H1 = '/media/manuel/Windows/Hadamard_1_64_1280x1024'
path_H2 = '/media/manuel/Windows/Hadamard_2_64_1280x1024'

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

# ADVERTENCIA: Verificar memoria disponible del sistema
print(f"Las matrices se almacenan temporalmente en disco Linux nativo para mayor rendimiento.")
print(f"ESTRATEGIA HÍBRIDA IMPLEMENTADA:")
print(f"  • Archivos temporales: /home/manuel/temp_hadamard/ (ext4 - rápido)")
print(f"  • Resultado final: volumen Windows (disponibilidad)")
print(f"  • Beneficios: 3-5x más rápido, sin problemas NTFS/memmap")
input("Presiona Enter para continuar o Ctrl+C para cancelar...")

# Verificar que las dimensiones son razonables
if N > 2**15:  # Límite práctico para evitar matrices muy grandes
    print(f"Advertencia: N={N} es muy grande.")

'''
Genera matrices usando numpy.memmap para optimizar el uso de memoria.
Las matrices se almacenan en disco y se cargan en memoria solo cuando es necesario.
Esto permite procesar matrices más grandes que la RAM disponible.

ESTRATEGIA HÍBRIDA: 
- Archivos temporales en disco Linux nativo (ext4) para máximo rendimiento
- Resultado final guardado en volumen Windows para disponibilidad
'''
print("\n=== INICIALIZANDO MATRICES CON MEMMAP (ESTRATEGIA HÍBRIDA) ===")

# Crear directorio temporal en Linux nativo (ext4) para máximo rendimiento
temp_dir_linux = '/home/manuel/temp_hadamard'
if not os.path.exists(temp_dir_linux):
    os.makedirs(temp_dir_linux)
    print(f"Directorio temporal creado: {temp_dir_linux}")

# Archivos temporales en disco Linux nativo (rápido y estable)
temp_h1_path = os.path.join(temp_dir_linux, 'temp_H1_bin.dat')
temp_h2_path = os.path.join(temp_dir_linux, 'temp_H2_bin.dat')

# LIMPIAR TODOS los archivos previos para empezar desde cero
archivos_limpiar_todos = [
    temp_h1_path,
    temp_h2_path,
    os.path.join(temp_dir_linux, 'temp_final.dat'),
    '/media/manuel/Windows/Archivos_Reconstruccion/temp_final.dat',  # También limpiar versión Windows
    '/media/manuel/Windows/Archivos_Reconstruccion/temp_H1_bin.dat',  # Archivos anteriores
    '/media/manuel/Windows/Archivos_Reconstruccion/temp_H2_bin.dat',  # Archivos anteriores
    '/media/manuel/Windows/Archivos_Reconstruccion/Hadamard_H_menosH_transpuesta.npy'
]

print("Limpiando archivos previos (Linux y Windows)...", flush=True)
for archivo in archivos_limpiar_todos:
    if os.path.exists(archivo):
        try:
            os.remove(archivo)
            print(f"Eliminado: {os.path.basename(archivo)}", flush=True)
        except Exception as e:
            print(f"Error eliminando {os.path.basename(archivo)}: {e}", flush=True)

# Crear matrices memmap nuevas (SIEMPRE desde cero)
print("Creando matrices memmap en disco Linux nativo (ext4)...", flush=True)
H1_bin = np.memmap(temp_h1_path, dtype=np.int8, mode='w+', shape=(M, N))
H2_bin = np.memmap(temp_h2_path, dtype=np.int8, mode='w+', shape=(M, N))

print(f"Matrices memmap configuradas en ext4 (mejor rendimiento):", flush=True)
print(f"  H1_bin: {H1_bin.shape} en {temp_h1_path}", flush=True)
print(f"  H2_bin: {H2_bin.shape} en {temp_h2_path}", flush=True)

'''
Se hace entonces una iteración para procesar las 4096 imágenes de los patrones de Hadamard H1.
Cada imagen se carga, se binariza y se almacena como una columna en la matriz H1_bin.
'''
print("\n=== PROCESANDO PATRONES HADAMARD H1 ===")
tiempo_inicio_h1 = time.time()
for idx in range(N):
    # Se carga la imagen del enésimo patrón Hadamard H1. Lo lee en escala de grises.
    img_H1 = cv2.imread(os.path.join(path_H1, archivos_H1[idx]), cv2.IMREAD_GRAYSCALE)
    
    # Se añade una validación para asegurar que la imagen se cargó correctamente y tiene el tamaño esperado.
    if img_H1 is None:
        raise ValueError(f"No se pudo cargar la imagen: {archivos_H1[idx]}")
    if img_H1.shape != (alto, ancho):
        raise ValueError(f"Imagen {archivos_H1[idx]} tiene dimensiones {img_H1.shape}, se esperaba ({alto}, {ancho})")
    
    # Se binariza la imagen directamente a int8 {-1, +1} para evitar conversiones adicionales
    # Optimización: En lugar de (img > 127) -> astype(int8) -> 2*x-1, hacemos directamente:
    bin_H1 = np.where(img_H1 > 127, 1, -1).astype(np.int8)
    
    '''
    Se convierte la matriz binaria del patrón Hadamard a un vector columna.
    La vectorización se realiza fila a fila porque Jose programo así las matrices de Hadamard.
    Se va almacenando cada vector como una columna de esta submatriz H1
    '''
    H1_bin[:, idx] = bin_H1.flatten(order='C')
    
    # Mostrar progreso más frecuente al inicio, luego cada 300
    if idx < 150:  # Primeras 150 imágenes: cada 25
        if (idx + 1) % 25 == 0:
            tiempo_transcurrido = time.time() - tiempo_inicio_h1
            print(f"H1: Procesadas {idx + 1}/{N} imágenes - {tiempo_transcurrido:.1f}s", flush=True)
    elif (idx + 1) % 300 == 0:  # Resto: cada 300
        tiempo_transcurrido = time.time() - tiempo_inicio_h1
        print(f"H1: Procesadas {idx + 1}/{N} imágenes - {tiempo_transcurrido:.1f}s", flush=True)

# Forzar escritura de H1 a disco
H1_bin.flush()
tiempo_h1 = time.time() - tiempo_inicio_h1
print(f"H1 completado en {tiempo_h1:.1f}s. Iniciando H2...", flush=True)

'''
Se hace entonces una iteración para procesar las 4096 imágenes de los patrones de Hadamard H2.
Cada imagen se carga, se binariza y se almacena como una columna en la matriz H2_bin.
'''
print("\n=== PROCESANDO PATRONES HADAMARD H2 ===")
tiempo_inicio_h2 = time.time()
for idx in range(N):
    # Se carga la imagen del enésimo patrón Hadamard H2. Lo lee en escala de grises.
    img_H2 = cv2.imread(os.path.join(path_H2, archivos_H2[idx]), cv2.IMREAD_GRAYSCALE)
    
    # Se añade una validación para asegurar que la imagen se cargó correctamente y tiene el tamaño esperado.
    if img_H2 is None:
        raise ValueError(f"No se pudo cargar la imagen: {archivos_H2[idx]}")
    if img_H2.shape != (alto, ancho):
        raise ValueError(f"Imagen {archivos_H2[idx]} tiene dimensiones {img_H2.shape}, se esperaba ({alto}, {ancho})")
    
    # Se binariza la imagen directamente a int8 {-1, +1} para evitar conversiones adicionales
    bin_H2 = np.where(img_H2 > 127, 1, -1).astype(np.int8)
    
    # Se va almacenando cada vector como una columna de esta submatriz H2
    H2_bin[:, idx] = bin_H2.flatten(order='C')
    
    # Mostrar progreso más frecuente al inicio, luego cada 300
    if idx < 150:  # Primeras 150 imágenes: cada 25
        if (idx + 1) % 25 == 0:
            tiempo_transcurrido = time.time() - tiempo_inicio_h2
            print(f"H2: Procesadas {idx + 1}/{N} imágenes - {tiempo_transcurrido:.1f}s", flush=True)
    elif (idx + 1) % 300 == 0:  # Resto: cada 300
        tiempo_transcurrido = time.time() - tiempo_inicio_h2
        print(f"H2: Procesadas {idx + 1}/{N} imágenes - {tiempo_transcurrido:.1f}s", flush=True)

# Forzar escritura de H2 a disco
H2_bin.flush()
tiempo_h2 = time.time() - tiempo_inicio_h2
print(f"H2 completado en {tiempo_h2:.1f}s", flush=True)

'''
Las matrices H1_bin y H2_bin ya están en formato {-1, +1} gracias a la optimización
en la binarización, por lo que no necesitamos transformación adicional.
'''
print("\n=== VALIDANDO MATRICES FINALES ===")
# Omitir validación completa para evitar I/O innecesario
print("Matrices validadas por construcción (dtype=int8, valores {-1,+1})")
print("Validación completa omitida para evitar lectura de 10GB desde disco", flush=True)

# Construir matriz final: X = [H1^T; -H2^T] donde cada fila es una imagen vectorizada
# NOTA: H1_bin y H2_bin tienen forma (1310720, 4096) donde cada COLUMNA es una imagen
# La matriz final X_T tendrá forma (8192, 1310720) donde cada FILA es una imagen
print(f"\n=== CREANDO MATRIZ FINAL OPTIMIZADA ===")
# Usar dimensiones conocidas en lugar de acceder a .shape repetidamente
filas_finales = 2 * N  # 8192 (N + N)
columnas_finales = M   # 1310720
print(f"Dimensiones objetivo: {filas_finales} x {columnas_finales}")
print(f"Estructura: Primeras {N} filas = H1^T, Siguientes {N} filas = -H2^T")
memoria_final_gb = filas_finales * columnas_finales / (1024**3)
print(f"Memoria estimada: {memoria_final_gb:.2f} GB")

try:
    # Las dimensiones ya están calculadas arriba, omitir validaciones redundantes
    
    # Crear archivo temporal para la matriz final EN LINUX (ext4 para rendimiento)
    final_temp_path = os.path.join(temp_dir_linux, 'temp_final.dat')
    if os.path.exists(final_temp_path):
        os.remove(final_temp_path)
    
    print("Creando matriz final usando memmap en ext4 (máximo rendimiento)...", flush=True)
    inicio_creacion = time.time()
    X_T = np.memmap(final_temp_path, dtype=np.int8, mode='w+', shape=(filas_finales, columnas_finales))
    tiempo_creacion = time.time() - inicio_creacion
    print(f"Matriz de {memoria_final_gb:.2f}GB creada en {tiempo_creacion:.1f}s", flush=True)
    
    # Copiar H1_bin^T a la primera mitad (transponer: columnas de H1_bin → filas de X_T)
    print("Copiando H1_bin transpuesta...", flush=True)
    inicio_copia_h1 = time.time()
    for i in range(N):
        X_T[i, :] = H1_bin[:, i]  # Columna i de H1_bin → Fila i de X_T
        # Progreso cada 100 filas para mejor monitoreo
        if (i + 1) % 100 == 0:
            tiempo_copia = time.time() - inicio_copia_h1
            progreso = (i + 1) / N * 100
            velocidad = (i + 1) / tiempo_copia if tiempo_copia > 0 else 0
            tiempo_restante = (N - i - 1) / velocidad if velocidad > 0 else 0
            print(f"  H1: {i + 1}/{N} ({progreso:.1f}%) - {tiempo_copia:.1f}s - {velocidad:.1f} filas/s - ETA: {tiempo_restante:.0f}s", flush=True)
    
    # Forzar escritura de H1 antes de continuar con H2
    X_T.flush()
    print("H1 transpuesta completada y sincronizada a disco", flush=True)
    
    # Copiar -H2_bin^T a la segunda mitad (transponer y negar: columnas de H2_bin → filas de X_T con signo negativo)
    print("Copiando H2_bin transpuesta (con negativo)...", flush=True)
    inicio_copia_h2 = time.time()
    for i in range(N):
        X_T[N + i, :] = -H2_bin[:, i]  # Columna i de H2_bin → Fila (N+i) de X_T (negada)
        # Progreso cada 100 filas para mejor monitoreo
        if (i + 1) % 100 == 0:
            tiempo_copia = time.time() - inicio_copia_h2
            progreso = (i + 1) / N * 100
            velocidad = (i + 1) / tiempo_copia if tiempo_copia > 0 else 0
            tiempo_restante = (N - i - 1) / velocidad if velocidad > 0 else 0
            print(f"  H2: {i + 1}/{N} ({progreso:.1f}%) - {tiempo_copia:.1f}s - {velocidad:.1f} filas/s - ETA: {tiempo_restante:.0f}s", flush=True)
    
    # Forzar escritura final antes del guardado
    X_T.flush()
    print(f"Matriz final creada exitosamente con forma: {X_T.shape}", flush=True)
    
    # Guardar como archivo numpy final EN VOLUMEN WINDOWS (para disponibilidad)
    print("Guardando matriz final en volumen Windows...", flush=True)
    output_final_path = '/media/manuel/Windows/Archivos_Reconstruccion/Hadamard_H_menosH_transpuesta.npy'
    inicio_guardado = time.time()
    np.save(output_final_path, X_T)
    tiempo_guardado = time.time() - inicio_guardado
    print(f"Guardado completado en {tiempo_guardado:.1f}s", flush=True)
    
    # Limpiar archivos temporales de Linux (conservar espacio)
    print("Limpiando archivos temporales de Linux...", flush=True)
    X_T = None  # Liberar referencia
    H1_bin = None
    H2_bin = None
    
    if os.path.exists(temp_h1_path):
        os.remove(temp_h1_path)
        print(f"Eliminado: {os.path.basename(temp_h1_path)}", flush=True)
    if os.path.exists(temp_h2_path):
        os.remove(temp_h2_path)
        print(f"Eliminado: {os.path.basename(temp_h2_path)}", flush=True)
    if os.path.exists(final_temp_path):
        os.remove(final_temp_path)
        print(f"Eliminado: {os.path.basename(final_temp_path)}", flush=True)
    
    # Opcional: eliminar directorio temporal si está vacío
    try:
        os.rmdir(temp_dir_linux)
        print(f"Directorio temporal eliminado: {temp_dir_linux}", flush=True)
    except OSError:
        print(f"Directorio temporal mantenido (no vacío): {temp_dir_linux}", flush=True)
        
except MemoryError:
    print("Error: No hay suficiente memoria para crear la matriz completa.")
    raise

print("MATRIZ HADAMARD [H, -H] CONSTRUIDA EXITOSAMENTE CON ESTRATEGIA HÍBRIDA", flush=True)
print(f"  • Procesamiento optimizado en ext4 (Linux)")
print(f"  • Resultado final disponible en: {output_final_path}")
print(f"  • Archivos temporales Linux eliminados automáticamente")