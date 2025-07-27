import numpy as np
import os
import time

'''
Este código construye la matriz de intensidad final a partir de los speckles vectorizados.
Carga las matrices H1 y H2 (cada una de 1310720 x 4096), aplica la fórmula 2*I^p - I^1,
y concatena horizontalmente para generar una matriz final de 1310720 x 8192.
'''

######################### Construcción de matriz de intensidad #########################

print("=== CONSTRUCCIÓN DE MATRIZ DE INTENSIDAD ===")
inicio_tiempo = time.time()
print("Iniciando carga de matrices de speckles vectorizados...")

# Configurar paths - OPTIMIZADO: usar disco local para todo el procesamiento
base_path = '/home/manuel/temp_intensity'  # Todo en disco local (más rápido)
temp_path = '/home/manuel/temp_intensity'  # Mismo directorio para eficiencia

# Crear directorio temporal si no existe
os.makedirs(temp_path, exist_ok=True)

# Cargar matrices de speckles vectorizados con memmap (optimización crítica de RAM)
print("Cargando matrices de speckles vectorizados con memmap...")
try:
    # Usar memmap para evitar cargar 10+ GB en RAM
    H1_speckles = np.load(f'{base_path}/speckles_H1_vectorizados.npy', mmap_mode='r')  
    H2_speckles = np.load(f'{base_path}/speckles_H2_vectorizados.npy', mmap_mode='r')  
    print("Matrices cargadas exitosamente desde disco local con memmap (ultra-optimizado)")
except FileNotFoundError as e:
    raise FileNotFoundError(f"No se pudo cargar el archivo: {e}")
except Exception as e:
    raise Exception(f"Error al cargar matrices: {e}")

# Solo verificaciones esenciales (las detalladas ya están en Matrix_Checks.py)
print(f"H1_speckles: {H1_speckles.shape}, {H1_speckles.dtype}")
print(f"H2_speckles: {H2_speckles.shape}, {H2_speckles.dtype}")

# Constantes esperadas basadas en validación previa
M_esperado, N_esperado = H1_speckles.shape


print(f"\n=== PROCESAMIENTO DE MATRIZ DE INTENSIDAD ===")

print("Aplicando fórmula: Y = 2 * I^p - I^1...")

# PROCESAMIENTO POR CHUNKS PARA EVITAR OVERFLOW DE RAM
# Se procesará columna por columna para mantener bajo uso de memoria RAM
print("Configurando procesamiento por chunks para optimizar uso de RAM...")

# Configuración de chunks - OPTIMIZACIÓN INTELIGENTE con límites de RAM
'''
================================ CONSIDERACIONES SOBRE chunk_size ================================

1. CONFIGURACIÓN ACTUAL:
   - Se utiliza un valor de `chunk_size = 1024`, equivalente a procesar 1/4 de todas las columnas
     (4096 columnas en total).
   - Esto implica un uso de memoria estimado de:
       RAM ≈ 1024 × 1310720 × 2 bytes × 2 matrices = ~5.37 GB
   - Esto respeta la RAM física disponible (7.6 GB), manteniendo margen de seguridad.

2. MOTIVACIÓN:
   - Esta configuración reduce el número de ciclos de procesamiento (solo 4 chunks)
   - Se mejora el throughput general reduciendo operaciones I/O intermedias.
   - Mantiene estabilidad del sistema sin forzar swap masivo.

3. VENTAJAS:
   - Comportamiento predecible en entornos con RAM limitada
   - No compite agresivamente con el sistema operativo por recursos
   - Permite operaciones concurrentes sin degradación significativa

4. ESCALABILIDAD:
   - Para máquinas con más RAM: chunk_size puede aumentarse proporcionalmente
   - Para máquinas con menos RAM: chunk_size se puede reducir a 512 o 256

===============================================================================================
'''
chunk_size = 512  # Chunk size balanceado: rendimiento + estabilidad
total_columnas = N_esperado  # 4096
num_chunks = (total_columnas + chunk_size - 1) // chunk_size  # División hacia arriba

# Calcular RAM estimada por chunk
ram_por_chunk_gb = chunk_size * M_esperado * 2 / (1024**3)  # int16 = 2 bytes por elemento

print(f"Configuración de chunks:")
print(f"- Tamaño de chunk: {chunk_size} columnas")
print(f"- Total de chunks: {num_chunks}")
print(f"- RAM estimada por chunk: {ram_por_chunk_gb:.3f} GB")

# Crear archivos de matrices temporales en disco usando memmap
print("Creando matrices temporales en disco...")
Y_H1_memmap = np.memmap(os.path.join(temp_path, 'temp_Y_H1.dat'), 
                        dtype=np.int16, mode='w+', shape=(M_esperado, N_esperado))
Y_H2_memmap = np.memmap(os.path.join(temp_path, 'temp_Y_H2.dat'), 
                        dtype=np.int16, mode='w+', shape=(M_esperado, N_esperado))

# Extraer I1 (vector columna base) y convertir a int16 de una vez
print("Extrayendo vector I1 (patrón Hadamard con todos los píxeles en 1)...")
I1 = H1_speckles[:, 0].reshape(-1, 1).astype(np.int16)
print(f"Vector I1 extraído: forma={I1.shape}, dtype={I1.dtype}")

# Pre-calcular índices de chunks para optimización
print("Pre-calculando índices de chunks...")
chunk_indices = [(i * chunk_size, min((i + 1) * chunk_size, total_columnas)) 
                 for i in range(num_chunks)]

# Procesar por chunks para evitar overflow de RAM
print(f"\n=== PROCESAMIENTO POR CHUNKS ===")
for i, (inicio, fin) in enumerate(chunk_indices):
    columnas_chunk = fin - inicio
    
    print(f"Procesando chunk {i+1}/{num_chunks}: columnas {inicio} a {fin-1} ({columnas_chunk} columnas)")
    
    # Aplicar fórmula vectorizada: Y = 2 * I^p - I^1
    # Carga y conversión directa para eficiencia
    Y_H1_memmap[:, inicio:fin] = 2 * H1_speckles[:, inicio:fin].astype(np.int16) - I1
    Y_H2_memmap[:, inicio:fin] = 2 * H2_speckles[:, inicio:fin].astype(np.int16) - I1
    
    # Mostrar progreso solo cada chunk (son pocos ahora)
    progreso = (i + 1) / num_chunks * 100
    print(f"Progreso: {progreso:.1f}% completado ({i+1}/{num_chunks})")

# Validación rápida solo en muestra pequeña
print(f"Procesamiento por chunks completado")
print(f"Y_H1_memmap: {Y_H1_memmap.shape}, {Y_H1_memmap.dtype}")
print(f"Y_H2_memmap: {Y_H2_memmap.shape}, {Y_H2_memmap.dtype}")

# Dimensiones para concatenación final
filas_finales = M_esperado
columnas_finales = 2 * N_esperado  # H1 + H2 concatenadas horizontalmente

# Concatenar horizontalmente usando memmap optimizado
print(f"\n=== CONCATENACIÓN FINAL CON MEMMAP ===")

# Calcular memoria final
bytes_por_elemento = 2  # int16
memoria_final_gb = filas_finales * columnas_finales * bytes_por_elemento / (1024**3)
print(f"Matriz final: {filas_finales} x {columnas_finales} ({memoria_final_gb:.2f} GB)")

try:
    # Crear matriz final memmap en disco local (más rápido)
    Matriz_Intensidad_memmap = np.memmap(os.path.join(base_path, 'temp_Matriz_Final.dat'), 
                                        dtype=np.int16, mode='w+', shape=(filas_finales, columnas_finales))
    
    # Concatenar por chunks optimizado
    print("Copiando Y_H1 y Y_H2 a matriz final...")
    chunk_concatenacion = 1500  # Chunks optimizados
    for i in range(0, filas_finales, chunk_concatenacion):
        fin_fila = min(i + chunk_concatenacion, filas_finales)
        # Operación vectorizada de concatenación
        Matriz_Intensidad_memmap[i:fin_fila, :N_esperado] = Y_H1_memmap[i:fin_fila, :]
        Matriz_Intensidad_memmap[i:fin_fila, N_esperado:] = Y_H2_memmap[i:fin_fila, :]
        
        # Progreso cada 100 chunks
        if (i // chunk_concatenacion + 1) % 100 == 0:
            progreso_concat = (fin_fila / filas_finales) * 100
            print(f"Concatenación: {progreso_concat:.1f}% completada")
    
    print(f"Concatenación exitosa: {Matriz_Intensidad_memmap.shape}")
    
except MemoryError:
    print("Error: No hay suficiente memoria para crear la matriz concatenada.")
    raise

# Guardar la matriz de intensidad final
print(f"\n=== GUARDADO DE RESULTADO ===")

try:
    # Uso directo de np.save con memmap - más eficiente
    archivo_final = os.path.join(base_path, 'Matriz_Intensidad.npy')
    print(f"Guardando en: {archivo_final}")
    
    # NumPy puede manejar memmap directamente
    np.save(archivo_final, Matriz_Intensidad_memmap)
    
    # Verificación de guardado
    if os.path.exists(archivo_final):
        tamaño_archivo_mb = os.path.getsize(archivo_final) / (1024**2)
        print(f"Archivo guardado: {tamaño_archivo_mb:.2f} MB")
    else:
        raise FileNotFoundError("El archivo no se guardó correctamente")
        
except Exception as e:
    print(f"Error al guardar: {e}")
    raise

# Limpiar archivos temporales
print(f"\n=== LIMPIEZA DE ARCHIVOS TEMPORALES ===")
try:
    # Lista de archivos temporales a eliminar
    temp_files = [
        os.path.join(temp_path, 'temp_Y_H1.dat'),
        os.path.join(temp_path, 'temp_Y_H2.dat'),
        os.path.join(base_path, 'temp_Matriz_Final.dat')
    ]
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Archivo temporal eliminado: {os.path.basename(temp_file)}")
    
    print("Limpieza completada")
except Exception as e:
    print(f"Advertencia: No se pudieron eliminar archivos temporales: {e}")

# Estadísticas finales con muestreo eficiente
print("\n=== ESTADÍSTICAS FINALES ===")
try:
    # Cargar con mmap para verificación final eficiente
    archivo_final = os.path.join(base_path, 'Matriz_Intensidad.npy')
    matriz_final = np.load(archivo_final, mmap_mode='r')
    
    print(f"Matriz final verificada: {matriz_final.shape}, {matriz_final.dtype}")
    
    # Muestreo estratificado eficiente - diferentes zonas de la matriz
    filas_total, cols_total = matriz_final.shape
    # Tomar muestras de 3 regiones (centro y esquinas)
    regiones = [
        (0, 0, 50, 50),  # Superior izquierda
        (filas_total//2-25, cols_total//2-25, 50, 50),  # Centro
        (filas_total-50, cols_total-50, 50, 50)  # Inferior derecha
    ]
    
    print("Verificación por muestreo:")
    for i, (fila_ini, col_ini, filas, cols) in enumerate(regiones):
        muestra_region = matriz_final[fila_ini:fila_ini+filas, col_ini:col_ini+cols]
        print(f"  Región {i+1}: rango=[{muestra_region.min()}, {muestra_region.max()}]")
    
except Exception as e:
    print(f"Error en estadísticas finales: {e}")

print(f"\nPROCESO COMPLETADO EXITOSAMENTE")
print(f"Matriz de intensidad guardada en: {os.path.join(base_path, 'Matriz_Intensidad.npy')}")
tiempo_final = time.time()
tiempo_total = tiempo_final - inicio_tiempo
print(f"Tiempo total de procesamiento: {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} minutos)")


