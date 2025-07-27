import numpy as np
import os
import time

'''
Script dedicado para concatenación final de matrices de intensidad.
Aprovecha los archivos temporales Y_H1 y Y_H2 ya procesados para crear la matriz final.
Optimizado para manejar matrices de 20+ GB con chunks ultra-pequeños.

ESTRATEGIA DE CONCATENACIÓN SEGURA:
- Parte de matrices temporales verificadas (temp_Y_H1.dat y temp_Y_H2.dat)
- Usa chunks ultra-pequeños (250 filas) para evitar bus errors
- Crea una matriz final completamente nueva desde cero
- No depende de archivos previos potencialmente corruptos

================================= NOTA SOBRE ARCHIVOS .dat vs .npy =================================

Durante la construcción de la matriz de intensidad, se utiliza un archivo temporal 
llamado `temp_Matriz_Final.dat` en formato binario plano (`memmap`) para almacenar 
los datos en disco sin cargarlos completamente en RAM. Este archivo contiene la matriz 
de tamaño (1310720, 8192) con datos tipo int16, y fue generado de forma incremental 
a partir de las submatrices Y_H1 y Y_H2.

El paso final opcional consiste en guardar esta matriz como `.npy` para mayor portabilidad 
y compatibilidad con herramientas de terceros. Sin embargo, este paso requiere tener 
la matriz completa ya construida, y en algunos casos puede duplicar el uso de disco 
(≈ 20 GB `.dat` + ≈ 20 GB `.npy`).

Dado esto, se plantea la siguiente solución:

1. Trabajar directamente con el archivo `.dat` mediante `np.memmap()`:
   - Más eficiente para análisis o reconstrucción por partes.
   - Requiere conocer y declarar explícitamente el `shape` y `dtype`.

2. El archivo `.npy` solo debe generarse si:
   - Se desea compartir el archivo.
   - Se necesita portabilidad sin necesidad de metadata externa.
   - Se trabaja con librerías que no manejan memmap plano.

Conclusión:
✔️ Para uso local y procesamiento eficiente, se recomienda **mantener solo el `.dat`**.
❗ Si se necesita portabilidad o distribución, se puede ejecutar un script adicional para convertirlo a `.npy`.

Ejemplo de carga del `.dat`:
    matriz = np.memmap('/ruta/temp_Matriz_Final.dat', dtype=np.int16, mode='r', shape=(1310720, 8192))

=========================================================================================
'''

print("=== CONCATENACIÓN FINAL OPTIMIZADA ===")
inicio_tiempo = time.time()

# Configurar paths
base_path = '/home/manuel/temp_intensity'
temp_path = '/home/manuel/temp_intensity'

# Verificar que existen los archivos temporales
archivos_requeridos = [
    os.path.join(temp_path, 'temp_Y_H1.dat'),
    os.path.join(temp_path, 'temp_Y_H2.dat')
]

print("Verificando archivos temporales...")
for archivo in archivos_requeridos:
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"Archivo temporal no encontrado: {archivo}")
    tamaño_gb = os.path.getsize(archivo) / (1024**3)
    print(f"- {os.path.basename(archivo)}: {tamaño_gb:.2f} GB")

# Constantes de las matrices (basadas en el procesamiento previo)
M_esperado = 1310720  # Filas
N_esperado = 4096     # Columnas por matriz
filas_finales = M_esperado
columnas_finales = 2 * N_esperado  # H1 + H2 concatenadas

# Verificar espacio disponible en disco
import shutil
espacio_libre_gb = shutil.disk_usage(base_path).free / (1024**3)
tamaño_matriz_final_gb = filas_finales * columnas_finales * 2 / (1024**3)  # int16 = 2 bytes

print(f"\nVerificación de espacio:")
print(f"- Espacio libre: {espacio_libre_gb:.2f} GB")
print(f"- Matriz final estimada: {tamaño_matriz_final_gb:.2f} GB")

if espacio_libre_gb < tamaño_matriz_final_gb * 1.1:  # 10% de margen
    raise Exception(f"Espacio insuficiente. Necesario: {tamaño_matriz_final_gb:.2f} GB, Disponible: {espacio_libre_gb:.2f} GB")

print(f"\nDimensiones objetivo:")
print(f"- Matriz final: {filas_finales} x {columnas_finales}")
print(f"- Tamaño estimado: {filas_finales * columnas_finales * 2 / (1024**3):.2f} GB")

# Cargar matrices temporales con memmap (modo lectura)
print("\nCargando matrices temporales...")
Y_H1_memmap = np.memmap(os.path.join(temp_path, 'temp_Y_H1.dat'), 
                        dtype=np.int16, mode='r', shape=(M_esperado, N_esperado))
Y_H2_memmap = np.memmap(os.path.join(temp_path, 'temp_Y_H2.dat'), 
                        dtype=np.int16, mode='r', shape=(M_esperado, N_esperado))

print(f"Y_H1_memmap: {Y_H1_memmap.shape}, {Y_H1_memmap.dtype}")
print(f"Y_H2_memmap: {Y_H2_memmap.shape}, {Y_H2_memmap.dtype}")

# ESTRATEGIA DE CONCATENACIÓN POR CHUNKS MUY PEQUEÑOS
print(f"\n=== CONCATENACIÓN CON CHUNKS ULTRA-PEQUEÑOS ===")

# Crear matriz final memmap
print("Creando matriz final...")
try:
    # Crear en base_path (disco local)
    Matriz_Intensidad_memmap = np.memmap(os.path.join(base_path, 'temp_Matriz_Final.dat'), 
                                        dtype=np.int16, mode='w+', shape=(filas_finales, columnas_finales))
    print("Matriz final memmap creada exitosamente")
except Exception as e:
    print(f"Error creando matriz final: {e}")
    raise

# Concatenación con chunks ultra-pequeños para evitar Bus Error
chunk_filas = 250  # Chunks aún más pequeños para garantizar estabilidad
total_chunks = (filas_finales + chunk_filas - 1) // chunk_filas

print(f"Configuración de concatenación:")
print(f"- Chunk size: {chunk_filas} filas")
print(f"- Total chunks: {total_chunks}")
print(f"- RAM por chunk: {chunk_filas * columnas_finales * 2 / (1024**3):.3f} GB")

try:
    print("\nIniciando concatenación por chunks...")
    
    for i in range(0, filas_finales, chunk_filas):
        fin_fila = min(i + chunk_filas, filas_finales)
        filas_chunk = fin_fila - i
        
        # Copiar chunk de Y_H1 a la primera mitad de columnas
        Matriz_Intensidad_memmap[i:fin_fila, :N_esperado] = Y_H1_memmap[i:fin_fila, :]
        
        # Copiar chunk de Y_H2 a la segunda mitad de columnas  
        Matriz_Intensidad_memmap[i:fin_fila, N_esperado:] = Y_H2_memmap[i:fin_fila, :]
        
        # Flush para asegurar escritura a disco después de cada chunk
        Matriz_Intensidad_memmap.flush()
        
        # Progreso cada 100 chunks o al final
        chunk_actual = (i // chunk_filas) + 1
        if chunk_actual % 100 == 0 or fin_fila == filas_finales:
            progreso = (fin_fila / filas_finales) * 100
            tiempo_transcurrido = time.time() - inicio_tiempo
            print(f"Concatenación: {progreso:.1f}% completada ({chunk_actual}/{total_chunks}) - {tiempo_transcurrido:.1f}s")
    
    print(f"Concatenación exitosa: {Matriz_Intensidad_memmap.shape}")
    
    # Verificación inmediata del archivo creado
    archivo_creado = os.path.join(base_path, 'temp_Matriz_Final.dat')
    if os.path.exists(archivo_creado):
        tamaño_real_gb = os.path.getsize(archivo_creado) / (1024**3)
        print(f"Archivo creado: {tamaño_real_gb:.2f} GB")
        
        # Verificar que el tamaño sea correcto
        tamaño_esperado_gb = filas_finales * columnas_finales * 2 / (1024**3)
        if abs(tamaño_real_gb - tamaño_esperado_gb) > 0.1:  # Tolerancia de 100 MB
            print(f"ADVERTENCIA: Tamaño inesperado. Esperado: {tamaño_esperado_gb:.2f} GB")
    else:
        raise Exception("El archivo de matriz final no fue creado correctamente")
    
except Exception as e:
    print(f"Error durante concatenación: {e}")
    raise

# Archivo final está listo - no generar .npy por defecto
print(f"\n=== MATRIZ FINAL COMPLETADA ===")
print("Usando solo formato .dat (más eficiente para uso local)")
print("La matriz está lista para validación y uso directo")

# Validación robusta basada en Matrix_Checks.py
print(f"\n=== VALIDACIÓN ROBUSTA DE MATRIZ FINAL ===")
try:
    # Verificar usando temp_Matriz_Final.dat
    archivo_verificar = os.path.join(base_path, 'temp_Matriz_Final.dat')
    matriz_verificar = np.memmap(archivo_verificar, dtype=np.int16, mode='r', 
                                shape=(filas_finales, columnas_finales))
    
    print(f"Archivo verificado: {archivo_verificar}")
    print(f"Dimensiones: {matriz_verificar.shape} (filas × columnas)")
    print(f"Tipo de datos: {matriz_verificar.dtype}")
    
    # 1. Verificación de estructura (H1 | H2)
    print(f"\nVerificación de estructura H1|H2:")
    mitad_columnas = matriz_verificar.shape[1] // 2
    print(f"  - Columnas H1: 0 a {mitad_columnas-1}")
    print(f"  - Columnas H2: {mitad_columnas} a {matriz_verificar.shape[1]-1}")
    
    # 2. Muestreo estratificado por regiones
    print(f"\nAnálisis por regiones (muestreo estratificado):")
    
    # Definir regiones de muestreo
    regiones_analisis = [
        ("Superior izquierda", 0, 100, 0, 100),
        ("Centro", filas_finales//2-50, filas_finales//2+50, columnas_finales//2-50, columnas_finales//2+50),
        ("Inferior derecha", filas_finales-100, filas_finales, columnas_finales-100, columnas_finales)
    ]
    
    for nombre_region, fila_ini, fila_fin, col_ini, col_fin in regiones_analisis:
        muestra = matriz_verificar[fila_ini:fila_fin, col_ini:col_fin]
        
        print(f"  {nombre_region}:")
        print(f"    - Región: [{fila_ini}:{fila_fin}, {col_ini}:{col_fin}]")
        print(f"    - Rango: [{muestra.min()}, {muestra.max()}]")
        print(f"    - Media: {muestra.mean():.2f}")
        print(f"    - Valores únicos: {len(np.unique(muestra[:50, :50]))}")  # Submuestra pequeña
    
    # 3. Verificación de separación H1/H2
    print(f"\nVerificación de separación H1/H2:")
    
    # Muestrear columnas de H1 y H2 para comparar
    col_h1_sample = 100  # Columna de H1
    col_h2_sample = mitad_columnas + 100  # Columna equivalente de H2
    
    filas_muestra = slice(1000, 1100)  # 100 filas centrales
    
    h1_sample = matriz_verificar[filas_muestra, col_h1_sample]
    h2_sample = matriz_verificar[filas_muestra, col_h2_sample]
    
    print(f"  Muestra H1 (col {col_h1_sample}): rango=[{h1_sample.min()}, {h1_sample.max()}]")
    print(f"  Muestra H2 (col {col_h2_sample}): rango=[{h2_sample.min()}, {h2_sample.max()}]")
    print(f"  Correlación H1-H2: {np.corrcoef(h1_sample, h2_sample)[0,1]:.4f}")
    
    # 4. Verificación de integridad de datos
    print(f"\nVerificación de integridad:")
    
    # Verificar que no hay valores de relleno obviamente incorrectos
    valores_extremos = {
        'zeros': np.sum(matriz_verificar[::1000, ::1000] == 0),  # Muestreo sparse
        'max_values': matriz_verificar[::1000, ::1000].max(),
        'min_values': matriz_verificar[::1000, ::1000].min()
    }
    
    print(f"  Valores extremos (muestreo):")
    print(f"    - Ceros: {valores_extremos['zeros']} elementos")
    print(f"    - Máximo: {valores_extremos['max_values']}")
    print(f"    - Mínimo: {valores_extremos['min_values']}")
    
    # Determinar si la validación es exitosa
    rango_esperado = abs(valores_extremos['max_values'] - valores_extremos['min_values'])
    
    if rango_esperado > 10:  # Rango razonable para datos de intensidad
        print(f"  Validación EXITOSA: Rango de datos es adecuado ({rango_esperado})")
        validacion_exitosa = True
    else:
        print(f"  Advertencia: Rango de datos muy pequeño ({rango_esperado})")
        validacion_exitosa = False
        
    print(f"\nRESULTADO DE VALIDACIÓN: {'EXITOSA' if validacion_exitosa else 'CON ADVERTENCIAS'}")
        
except Exception as e:
    print(f"Error en validación: {e}")
    validacion_exitosa = False

print(f"\n=== RESUMEN FINAL ===")
tiempo_final = time.time()
tiempo_total = tiempo_final - inicio_tiempo
print(f"Tiempo de procesamiento: {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} minutos)")

# Información de archivos resultantes
archivo_dat = os.path.join(base_path, 'temp_Matriz_Final.dat')

print(f"\nARCHIVO PRINCIPAL:")
print(f"  Matriz de intensidad: {archivo_dat}")
try:
    tamaño_gb = os.path.getsize(archivo_dat) / (1024**3)
    print(f"  Tamaño: {tamaño_gb:.2f} GB")
except:
    print(f"  Tamaño: No disponible")

# Estado de validación
if 'validacion_exitosa' in locals():
    estado_validacion = "VALIDADA" if validacion_exitosa else "CON ADVERTENCIAS"
else:
    estado_validacion = "NO VALIDADA"

print(f"  Estado: {estado_validacion}")

print(f"\n=== INSTRUCCIONES DE USO ===")
print(f"Para cargar la matriz en Python:")
print(f"```python")
print(f"import numpy as np")
print(f"# Cargar matriz de intensidad (formato eficiente)")
print(f"matriz = np.memmap('{archivo_dat}', dtype=np.int16, mode='r', shape=(1310720, 8192))")
print(f"# Acceso a submatrices:")
print(f"h1_data = matriz[:, :4096]    # Matrices H1 procesadas")
print(f"h2_data = matriz[:, 4096:]    # Matrices H2 procesadas")
print(f"```")

print(f"\n=== INFORMACIÓN TÉCNICA ===")
print(f"Dimensiones: 1310720 × 8192 (filas × columnas)")
print(f"Tipo de datos: int16 (valores con signo de 16 bits)")
print(f"Estructura: [H1_matrices | H2_matrices] (concatenación horizontal)")
print(f"Fórmula aplicada: Y = 2*I^p - I^1 (para cada matriz)")
print(f"Uso recomendado: Reconstrucción de imágenes por método Hadamard")

if 'validacion_exitosa' in locals() and validacion_exitosa:
    print(f"\nPROCESO COMPLETADO EXITOSAMENTE")
    print(f"La matriz de intensidad está lista para usar")
else:
    print(f"\nPROCESO COMPLETADO CON ADVERTENCIAS")
    print(f"Revisar los resultados de validación antes de usar")
