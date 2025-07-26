import cv2
import numpy as np
import os

'''
Este código construye la matriz de intensidad final a partir de los speckles vectorizados.
Carga las matrices H1 y H2 (cada una de 1310720 x 4096), aplica la fórmula 2*I^p - I^1,
y concatena horizontalmente para generar una matriz final de 1310720 x 8192.
'''

######################### Construcción de matriz de intensidad #########################

print("=== CONSTRUCCIÓN DE MATRIZ DE INTENSIDAD ===")
print("Iniciando carga de matrices de speckles vectorizados...")

# Cargar matrices de speckles vectorizados
try:
    H1_speckles = np.load('D:\\Archivos_Reconstruccion\\speckles_H1_vectorizados.npy')  
    H2_speckles = np.load('D:\\Archivos_Reconstruccion\\speckles_H2_vectorizados.npy')  
    print("Matrices cargadas exitosamente")
except FileNotFoundError as e:
    raise FileNotFoundError(f"No se pudo cargar el archivo: {e}")
except Exception as e:
    raise Exception(f"Error al cargar matrices: {e}")

# Validaciones dimensionales y de tipos
print(f"\n=== VALIDACIÓN DE MATRICES CARGADAS ===")
print(f"H1_speckles: forma={H1_speckles.shape}, dtype={H1_speckles.dtype}")
print(f"H2_speckles: forma={H2_speckles.shape}, dtype={H2_speckles.dtype}")

# Verificar dimensiones esperadas
M_esperado, N_esperado = 1310720, 4096
if H1_speckles.shape != (M_esperado, N_esperado):
    raise ValueError(f"H1_speckles tiene forma {H1_speckles.shape}, se esperaba ({M_esperado}, {N_esperado})")
if H2_speckles.shape != (M_esperado, N_esperado):
    raise ValueError(f"H2_speckles tiene forma {H2_speckles.shape}, se esperaba ({M_esperado}, {N_esperado})")

# Verificar tipos de datos consistentes
if H1_speckles.dtype != H2_speckles.dtype:
    print(f"Advertencia: Tipos de datos diferentes - H1: {H1_speckles.dtype}, H2: {H2_speckles.dtype}")

# Verificar rangos de valores
print(f"Rangos de valores:")
print(f"H1_speckles: [{H1_speckles.min()}, {H1_speckles.max()}]")
print(f"H2_speckles: [{H2_speckles.min()}, {H2_speckles.max()}]")

# Calcular memoria utilizada por las matrices cargadas
memoria_H1_gb = H1_speckles.nbytes / (1024**3)
memoria_H2_gb = H2_speckles.nbytes / (1024**3)
print(f"\nMemoria utilizada:")
print(f"H1_speckles: {memoria_H1_gb:.2f} GB")
print(f"H2_speckles: {memoria_H2_gb:.2f} GB")
print(f"Total matrices cargadas: {memoria_H1_gb + memoria_H2_gb:.2f} GB")


print(f"\n=== PROCESAMIENTO DE MATRIZ DE INTENSIDAD ===")

print("Aplicando fórmula: Y = 2 * I^p - I^1...")

# PROCESAMIENTO POR CHUNKS PARA EVITAR OVERFLOW DE RAM
# Se procesará columna por columna para mantener bajo uso de memoria RAM
print("Configurando procesamiento por chunks para optimizar uso de RAM...")

# Configuración de chunks
chunk_size = 100  # Procesar 100 columnas a la vez
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
Y_H1_memmap = np.memmap('D:\\Archivos_Reconstruccion\\temp_Y_H1.dat', 
                        dtype=np.int16, mode='w+', shape=(M_esperado, N_esperado))
Y_H2_memmap = np.memmap('D:\\Archivos_Reconstruccion\\temp_Y_H2.dat', 
                        dtype=np.int16, mode='w+', shape=(M_esperado, N_esperado))

# Extraer I1 (vector columna base) y convertir a int16 de una vez
print("Extrayendo vector I1 (patrón Hadamard con todos los píxeles en 1)...")
I1 = H1_speckles[:, 0].reshape(-1, 1).astype(np.int16)
print(f"Vector I1 extraído y convertido: forma={I1.shape}, rango=[{I1.min()}, {I1.max()}], dtype={I1.dtype}")

# Verificar que I1 tiene las dimensiones correctas
if I1.shape != (M_esperado, 1):
    raise ValueError(f"I1 tiene forma {I1.shape}, se esperaba ({M_esperado}, 1)")

# Procesar por chunks para evitar overflow de RAM
print(f"\n=== PROCESAMIENTO POR CHUNKS ===")
for i in range(num_chunks):
    inicio = i * chunk_size
    fin = min((i + 1) * chunk_size, total_columnas)
    columnas_chunk = fin - inicio
    
    print(f"Procesando chunk {i+1}/{num_chunks}: columnas {inicio} a {fin-1} ({columnas_chunk} columnas)")
    
    # Cargar chunk de H1 y convertir a int16
    H1_chunk = H1_speckles[:, inicio:fin].astype(np.int16)
    H2_chunk = H2_speckles[:, inicio:fin].astype(np.int16)
    
    # Aplicar fórmula: Y = 2 * I^p - I^1
    Y_H1_chunk = 2 * H1_chunk - I1
    Y_H2_chunk = 2 * H2_chunk - I1
    
    # Guardar chunk en memmap (esto escribe directamente al disco)
    Y_H1_memmap[:, inicio:fin] = Y_H1_chunk
    Y_H2_memmap[:, inicio:fin] = Y_H2_chunk
    
    # Limpiar memoria del chunk
    del H1_chunk, H2_chunk, Y_H1_chunk, Y_H2_chunk
    
    # Mostrar progreso
    if (i + 1) % 10 == 0 or i == num_chunks - 1:
        progreso = (i + 1) / num_chunks * 100
        print(f"Progreso: {progreso:.1f}% completado")

print("Procesamiento por chunks completado")

# Validar resultados de algunos chunks
print(f"\nValidando resultados:")
print(f"Y_H1_memmap: forma={Y_H1_memmap.shape}, dtype={Y_H1_memmap.dtype}")
print(f"Y_H2_memmap: forma={Y_H2_memmap.shape}, dtype={Y_H2_memmap.dtype}")

# Verificar una muestra pequeña de los datos
sample_cols = min(10, N_esperado)
print(f"Muestra de primeras {sample_cols} columnas:")
print(f"Y_H1 rango: [{Y_H1_memmap[:, :sample_cols].min()}, {Y_H1_memmap[:, :sample_cols].max()}]")
print(f"Y_H2 rango: [{Y_H2_memmap[:, :sample_cols].min()}, {Y_H2_memmap[:, :sample_cols].max()}]")

# Concatenar horizontalmente usando memmap
print(f"\n=== CONCATENACIÓN FINAL CON MEMMAP ===")
print("Creando matriz final concatenada en disco...")

# Verificar que las matrices tienen dimensiones compatibles para concatenación
if Y_H1_memmap.shape[0] != Y_H2_memmap.shape[0]:
    raise ValueError(f"Matrices incompatibles para concatenación: Y_H1 filas={Y_H1_memmap.shape[0]}, Y_H2 filas={Y_H2_memmap.shape[0]}")

# Calcular dimensiones finales antes de concatenar
filas_finales = Y_H1_memmap.shape[0]  # 1310720
columnas_finales = Y_H1_memmap.shape[1] + Y_H2_memmap.shape[1]  # 4096 + 4096 = 8192

print(f"Dimensiones de matriz final esperadas: ({filas_finales}, {columnas_finales})")

# Verificar overflow en el tamaño total de la matriz
elementos_totales = filas_finales * columnas_finales
if elementos_totales > np.iinfo(np.intp).max:
    raise ValueError(f"El tamaño de la matriz final ({filas_finales} x {columnas_finales}) "
                    f"excede el límite máximo de elementos en un array numpy")

# Calcular memoria estimada para la matriz final
bytes_por_elemento = Y_H1_memmap.itemsize  # tamaño en bytes del tipo de dato (int16 = 2 bytes)
memoria_final_gb = elementos_totales * bytes_por_elemento / (1024**3)

# Calcular uso eficiente de RAM (solo chunks pequeños)
ram_chunk_gb = chunk_size * M_esperado * bytes_por_elemento / (1024**3)
print(f"Memoria final en disco: {memoria_final_gb:.2f} GB")
print(f"RAM máxima utilizada: {ram_por_chunk_gb:.3f} GB por chunk de procesamiento")
print(f"Concatenando matrices por chunks para minimizar uso de RAM...")

try:
    # Crear matriz final memmap para concatenación
    Matriz_Intensidad_memmap = np.memmap('D:\\Archivos_Reconstruccion\\temp_Matriz_Final.dat', 
                                        dtype=np.int16, mode='w+', shape=(filas_finales, columnas_finales))
    
    # Concatenar por chunks para minimizar RAM
    print("Copiando Y_H1 a matriz final...")
    chunk_concatenacion = 500  # Procesar 500 filas a la vez
    for i in range(0, filas_finales, chunk_concatenacion):
        fin_fila = min(i + chunk_concatenacion, filas_finales)
        # Copiar H1 (primeras 4096 columnas)
        Matriz_Intensidad_memmap[i:fin_fila, :N_esperado] = Y_H1_memmap[i:fin_fila, :]
        # Copiar H2 (siguientes 4096 columnas)
        Matriz_Intensidad_memmap[i:fin_fila, N_esperado:] = Y_H2_memmap[i:fin_fila, :]
        
        if (i // chunk_concatenacion + 1) % 100 == 0:
            progreso_concat = (fin_fila / filas_finales) * 100
            print(f"Concatenación: {progreso_concat:.1f}% completada")
    
    print(f"Concatenación exitosa: forma={Matriz_Intensidad_memmap.shape}, dtype={Matriz_Intensidad_memmap.dtype}")
    
    # Verificar muestra de la matriz concatenada
    sample_size = min(1000, filas_finales)
    muestra = Matriz_Intensidad_memmap[:sample_size, :10]  # Primera muestra pequeña
    print(f"Rango de muestra final: [{muestra.min()}, {muestra.max()}]")
    
except MemoryError:
    print("Error: No hay suficiente memoria para crear la matriz concatenada.")
    raise

print("Concatenación terminada exitosamente :D")

# Guardar la matriz de intensidad final
print(f"\n=== GUARDADO DE RESULTADO ===")
print("Convirtiendo memmap a archivo .npy final...")

try:
    # Convertir memmap a array normal y guardar
    # Esto se hace por chunks para evitar cargar toda la matriz en RAM
    print("Guardando matriz final por chunks...")
    
    # Crear array temporal para chunks pequeños
    chunk_save = 1000  # Guardar 1000 filas a la vez
    
    # Inicializar archivo final
    with open('D:\\Archivos_Reconstruccion\\Matriz_Intensidad.npy', 'wb') as f:
        # Guardar header de numpy
        np.lib.format.write_array_header_1_0(f, 
            {'descr': np.dtype(np.int16).descr, 
             'fortran_order': False, 
             'shape': (filas_finales, columnas_finales)})
        
        # Guardar datos por chunks
        for i in range(0, filas_finales, chunk_save):
            fin_chunk = min(i + chunk_save, filas_finales)
            chunk_data = np.array(Matriz_Intensidad_memmap[i:fin_chunk, :])
            chunk_data.tofile(f)
            
            if (i // chunk_save + 1) % 100 == 0:
                progreso_save = (fin_chunk / filas_finales) * 100
                print(f"Guardado: {progreso_save:.1f}% completado")
    
    # Verificar que el archivo se guardó correctamente
    archivo_guardado = 'D:\\Archivos_Reconstruccion\\Matriz_Intensidad.npy'
    if os.path.exists(archivo_guardado):
        tamaño_archivo_mb = os.path.getsize(archivo_guardado) / (1024**2)
        tamaño_esperado_mb = (filas_finales * columnas_finales * 2) / (1024**2)  # 2 bytes por int16
        print(f"Archivo guardado exitosamente: {tamaño_archivo_mb:.2f} MB")
        print(f"Tamaño esperado: {tamaño_esperado_mb:.2f} MB")
        
        # Validar que el tamaño sea correcto
        diferencia_porcentaje = abs(tamaño_archivo_mb - tamaño_esperado_mb) / tamaño_esperado_mb * 100
        if diferencia_porcentaje > 5:  # Más del 5% de diferencia
            print(f"⚠️  Advertencia: Diferencia de tamaño {diferencia_porcentaje:.1f}%")
        else:
            print("✅ Tamaño de archivo validado correctamente")
    else:
        raise FileNotFoundError("El archivo no se guardó correctamente")
        
except Exception as e:
    print(f"Error al guardar: {e}")
    raise

# Limpiar archivos temporales
print(f"\n=== LIMPIEZA DE ARCHIVOS TEMPORALES ===")
try:
    import os
    temp_files = [
        'D:\\Archivos_Reconstruccion\\temp_Y_H1.dat',
        'D:\\Archivos_Reconstruccion\\temp_Y_H2.dat',
        'D:\\Archivos_Reconstruccion\\temp_Matriz_Final.dat'
    ]
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Archivo temporal eliminado: {temp_file}")
    
    print("Limpieza de archivos temporales completada")
except Exception as e:
    print(f"Advertencia: No se pudieron eliminar todos los archivos temporales: {e}")

# Calcular rangos finales con muestreo representativo para el resumen
print("Calculando estadísticas finales con muestreo distribuido...")
muestra_final = np.load('D:\\Archivos_Reconstruccion\\Matriz_Intensidad.npy', mmap_mode='r')

# MUESTREO REPRESENTATIVO: Tomar muestras de diferentes zonas de la matriz
# para evitar bias por zonas oscuras o artefactos localizados
filas_total, cols_total = muestra_final.shape

# Definir puntos de muestreo distribuidos uniformemente
num_muestras_fila = 10  # 10 zonas diferentes en filas
num_muestras_col = 20   # 20 zonas diferentes en columnas
tamaño_muestra_fila = 50  # 50 filas por muestra
tamaño_muestra_col = 20   # 20 columnas por muestra

print(f"Muestreando {num_muestras_fila}x{num_muestras_col} zonas distribuidas uniformemente...")

# Crear índices distribuidos uniformemente
indices_fila = np.linspace(0, filas_total - tamaño_muestra_fila, num_muestras_fila, dtype=int)
indices_col = np.linspace(0, cols_total - tamaño_muestra_col, num_muestras_col, dtype=int)

# Recopilar valores de todas las muestras distribuidas
valores_muestreados = []
total_muestras = len(indices_fila) * len(indices_col)
muestra_actual = 0

for i_fila in indices_fila:
    for i_col in indices_col:
        muestra_region = muestra_final[i_fila:i_fila+tamaño_muestra_fila, 
                                     i_col:i_col+tamaño_muestra_col]
        valores_muestreados.extend(muestra_region.flatten())
        
        muestra_actual += 1
        if muestra_actual % 50 == 0:  # Mostrar progreso cada 50 muestras
            progreso_muestra = (muestra_actual / total_muestras) * 100
            print(f"Muestreo: {progreso_muestra:.1f}% completado")

# Convertir a numpy array para cálculos estadísticos
valores_muestreados = np.array(valores_muestreados)
rango_min = valores_muestreados.min()
rango_max = valores_muestreados.max()
rango_medio = valores_muestreados.mean()
rango_std = valores_muestreados.std()

print(f"Estadísticas de {len(valores_muestreados):,} valores muestreados uniformemente:")
print(f"- Rango: [{rango_min}, {rango_max}]")
print(f"- Media: {rango_medio:.2f}")
print(f"- Desviación estándar: {rango_std:.2f}")

# Resumen final
print(f"\n=== RESUMEN FINAL ===")
print(f"Matriz de intensidad creada exitosamente")
print(f"Forma final: ({filas_finales}, {columnas_finales})")
print(f"Tipo de dato: int16")
print(f"Rango de valores (muestreo distribuido): [{rango_min}, {rango_max}]")
print(f"Media de valores: {rango_medio:.2f} ± {rango_std:.2f}")
print(f"Memoria de la matriz final en disco: {memoria_final_gb:.2f} GB")
print(f"RAM máxima utilizada durante procesamiento: {ram_por_chunk_gb:.3f} GB")
print(f"Archivo guardado en: D:\\Archivos_Reconstruccion\\Matriz_Intensidad.npy")


