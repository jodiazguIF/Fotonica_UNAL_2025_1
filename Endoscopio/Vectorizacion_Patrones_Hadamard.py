import numpy as np
import os
import time
import shutil
from scipy.linalg import hadamard

'''
Vectorización de patrones de Hadamard proyectados en el DMD.
Genera matriz Hadamard completa N²×N² y crea archivo comprimido para reconstrucción.
'''

print("=== GENERACIÓN DE PATRONES HADAMARD ===")

# Dimensiones DMD y parámetros
DMD_size = (1280, 1024)  # (ancho, alto)
ancho, alto = DMD_size
M = alto * ancho    # 1310720 pixels totales

# Parámetros Hadamard
pattern_size = 64  # Tamaño base 64x64
N2 = pattern_size ** 2  # 4096 patrones por conjunto
N = 2 * N2  # 8192 patrones totales (H1 + H2)

print(f"DMD: {ancho}×{alto} = {M} pixels")
print(f"Matriz Hadamard: {N2}×{N2}")
print(f"Patrón base: {pattern_size}×{pattern_size}")
print(f"Total patrones: H1({N2}) + H2({N2}) = {N}")

# Escalado y centrado para DMD
scale = min(DMD_size) // pattern_size  # 16x escalado
scaled_size = pattern_size * scale  # 1024 pixels escalados
offset_x = (ancho - scaled_size) // 2  # 128 offset horizontal
offset_y = (alto - scaled_size) // 2   # 0 offset vertical

print(f"Escalado: {scale}x (patrón {pattern_size}×{pattern_size} → {scaled_size}×{scaled_size})")
print(f"Centrado: offset_x={offset_x}, offset_y={offset_y}")

# Generar matriz Hadamard completa
print(f"Generando matriz Hadamard {N2}×{N2}...")
inicio_hadamard = time.time()
H_full = hadamard(N2, dtype=np.int8)  # Matriz completa 4096×4096
tiempo_hadamard = time.time() - inicio_hadamard
print(f"Matriz Hadamard generada en {tiempo_hadamard:.2f}s ({H_full.nbytes / (1024**3):.3f} GB)")

def make_hadamard_pattern_optimized(H_full, col_idx, pattern_size, DMD_size, scale, offset_x, offset_y):
    """
    Genera patrón Hadamard individual escalado y centrado para DMD
    
    Args:
        H_full: Matriz Hadamard completa N²×N²
        col_idx: Índice de columna a extraer
        pattern_size: Tamaño base (64)
        DMD_size: Dimensiones DMD (ancho, alto)
        scale: Factor de escalado
        offset_x, offset_y: Offsets de centrado
    
    Returns:
        np.array: Patrón vectorizado de tamaño M
    """
    # Extraer columna de matriz Hadamard
    columna = H_full[:, col_idx]
    
    # Reshape a matriz cuadrada 64×64
    square = columna.reshape((pattern_size, pattern_size), order='C')
    
    # Escalar patrón
    square_scaled = np.repeat(np.repeat(square, scale, axis=0), scale, axis=1)
    
    # Crear canvas DMD y centrar patrón
    canvas = np.zeros(DMD_size[::-1], dtype=np.int8)  # (alto, ancho)
    canvas[offset_y:offset_y+square_scaled.shape[0], offset_x:offset_x+square_scaled.shape[1]] = square_scaled
    
    # Vectorizar con orden C (row-major)
    return canvas.ravel(order='C')

# Calcular memoria requerida con compresión
memoria_individual_comprimida = M * N2 / 8 / (1024**3)
memoria_final_comprimida = N * M / 8 / (1024**3)
memoria_total_comprimida = 2 * memoria_individual_comprimida + memoria_final_comprimida

print(f"=== ESTIMACIÓN DE MEMORIA ===")
print(f"H1 comprimida: {memoria_individual_comprimida:.2f} GB ({N2} patrones)")
print(f"H2 comprimida: {memoria_individual_comprimida:.2f} GB ({N2} patrones)") 
print(f"Matriz final comprimida: {memoria_final_comprimida:.2f} GB ({N} × {M//8})")
print(f"TOTAL estimado: {memoria_total_comprimida:.2f} GB")

input("Presiona Enter para continuar...")

# Configurar almacenamiento temporal
temp_dir = '/home/manuel/temp_hadamard'
os.makedirs(temp_dir, exist_ok=True)
print(f"Directorio temporal: {temp_dir}")

# Limpiar archivos previos
archivos_limpiar = [
    os.path.join(temp_dir, 'H1_compressed.dat'),
    os.path.join(temp_dir, 'H2_compressed.dat'),
    os.path.join(temp_dir, 'X_T_final.dat'),
    '/media/manuel/Windows/Archivos_Reconstruccion/Hadamard_H_menosH_transpuesta_compressed.dat'
]

for archivo in archivos_limpiar:
    if os.path.exists(archivo):
        try:
            os.remove(archivo)
            print(f"Eliminado: {os.path.basename(archivo)}")
        except Exception as e:
            print(f"Error eliminando {os.path.basename(archivo)}: {e}")

# Crear matrices memmap comprimidas
temp_h1_path = os.path.join(temp_dir, 'H1_compressed.dat')
temp_h2_path = os.path.join(temp_dir, 'H2_compressed.dat')
compressed_M = M // 8  # Compresión 8:1 con packbits

H1_compressed = np.memmap(temp_h1_path, dtype=np.uint8, mode='w+', shape=(compressed_M, N2))
H2_compressed = np.memmap(temp_h2_path, dtype=np.uint8, mode='w+', shape=(compressed_M, N2))

print(f"Matrices memmap:")
print(f"  H1: {H1_compressed.shape} en {os.path.basename(temp_h1_path)}")
print(f"  H2: {H2_compressed.shape} en {os.path.basename(temp_h2_path)}")

print("\n=== GENERANDO H1 ===")
tiempo_inicio = time.time()

# Procesar en bloques para eficiencia
block_cols = 64
total_blocks = (N2 + block_cols - 1) // block_cols
print(f"Procesando {total_blocks} bloques de {block_cols} columnas")

for block_idx in range(total_blocks):
    start_col = block_idx * block_cols
    end_col = min(start_col + block_cols, N2)
    actual_cols = end_col - start_col
    
    # Generar bloque de patrones
    block_patterns = np.zeros((M, actual_cols), dtype=np.int8)
    
    for i, col_idx in enumerate(range(start_col, end_col)):
        patron_vector = make_hadamard_pattern_optimized(H_full, col_idx, pattern_size, DMD_size, scale, offset_x, offset_y)
        block_patterns[:, i] = patron_vector
    
    # Convertir a binario y comprimir
    bits_block = (block_patterns > 0)
    compressed_block = np.packbits(bits_block, axis=0)
    H1_compressed[:, start_col:end_col] = compressed_block
    
    # Progreso
    if (block_idx + 1) % 10 == 0 or block_idx == total_blocks - 1:
        progreso = (block_idx + 1) / total_blocks * 100
        tiempo_parcial = time.time() - tiempo_inicio
        velocidad = end_col / tiempo_parcial
        print(f"H1: {block_idx + 1}/{total_blocks} ({progreso:.1f}%) - {velocidad:.1f} col/s", flush=True)

H1_compressed.flush()
tiempo_h1 = time.time() - tiempo_inicio
print(f"H1 completado: {N2} patrones en {tiempo_h1:.1f}s")

print("\n=== GENERANDO H2 ===")
tiempo_inicio_h2 = time.time()

# H2 = -H1, reutilizar datos comprimidos
print("H2 = -H1: Copiando datos de H1...")
H2_compressed[:, :] = H1_compressed[:, :]
H2_compressed.flush()

tiempo_h2 = time.time() - tiempo_inicio_h2
print(f"H2 completado por copia en {tiempo_h2:.2f}s")

print("\n=== CONSTRUYENDO MATRIZ FINAL [H, -H]^T ===")

# Crear matriz final
temp_final_path = os.path.join(temp_dir, 'X_T_final.dat')
filas_finales = N  # 8192
columnas_finales_comprimidas = compressed_M

print(f"Matriz final: {filas_finales} × {columnas_finales_comprimidas}")
inicio_construccion = time.time()

X_T = np.memmap(temp_final_path, dtype=np.uint8, mode='w+', 
                shape=(filas_finales, columnas_finales_comprimidas))

# Construcción por bloques usando slicing
block_rows = 64
total_blocks = (N2 + block_rows - 1) // block_rows

print("Copiando H1 transpuesta...")
for block_idx in range(total_blocks):
    start_row = block_idx * block_rows
    end_row = min(start_row + block_rows, N2)
    
    # Leer bloque de H1 y transponer
    block_data = H1_compressed[:, start_row:end_row]
    X_T[start_row:end_row, :] = block_data.T
    
    if (block_idx + 1) % 20 == 0 or block_idx == total_blocks - 1:
        progreso = (block_idx + 1) / total_blocks * 100
        print(f"  H1: {progreso:.1f}%", flush=True)

print("Copiando H2 transpuesta...")
for block_idx in range(total_blocks):
    start_row = block_idx * block_rows
    end_row = min(start_row + block_rows, N2)
    
    # Reutilizar datos de H1 (negación al descomprimir)
    block_data = H1_compressed[:, start_row:end_row]
    X_T[N2 + start_row:N2 + end_row, :] = block_data.T
    
    if (block_idx + 1) % 20 == 0 or block_idx == total_blocks - 1:
        progreso = (block_idx + 1) / total_blocks * 100
        print(f"  H2: {progreso:.1f}%", flush=True)

X_T.flush()
tiempo_construccion = time.time() - inicio_construccion
print(f"Matriz final construida en {tiempo_construccion:.1f}s")

# Mover archivo final a destino
final_path_ntfs = '/media/manuel/Windows/Archivos_Reconstruccion/Hadamard_H_menosH_transpuesta_compressed.dat'
print(f"Moviendo archivo final...")

inicio_movimiento = time.time()
try:
    os.replace(temp_final_path, final_path_ntfs)
    tiempo_movimiento = time.time() - inicio_movimiento
    print(f"Archivo movido en {tiempo_movimiento:.1f}s")
except Exception as e:
    print(f"Error moviendo archivo: {e}")
    shutil.move(temp_final_path, final_path_ntfs)
    tiempo_movimiento = time.time() - inicio_movimiento
    print(f"Archivo copiado en {tiempo_movimiento:.1f}s")

# Limpiar archivos temporales
print("\nLimpiando archivos temporales...")
H1_compressed = None
H2_compressed = None
X_T = None

archivos_temporales = [temp_h1_path, temp_h2_path]
for archivo in archivos_temporales:
    if os.path.exists(archivo):
        os.remove(archivo)
        print(f"Eliminado: {os.path.basename(archivo)}")

try:
    os.rmdir(temp_dir)
    print(f"Directorio temporal eliminado")
except OSError:
    print(f"Directorio temporal mantenido (no vacío)")

# Resumen final
tiempo_total = time.time() - tiempo_inicio
tamaño_final = memoria_final_comprimida

print(f"\n=== GENERACIÓN COMPLETADA ===")
print(f"Archivo: {final_path_ntfs}")
print(f"Dimensiones: {filas_finales} × {columnas_finales_comprimidas} (comprimido)")
print(f"Tamaño: {tamaño_final:.1f} GB")
print(f"Tiempo total: {tiempo_total:.1f}s ({N/tiempo_total:.1f} pat/s)")
print(f"H1: {N2} patrones, H2: {N2} patrones")
print(f"Escalado: {scale}x, Centrado: ({offset_x}, {offset_y})")

print(f"\n=== DESCOMPRESIÓN ===")
print(f"# Cargar matriz")
print(f"X = np.memmap('{final_path_ntfs}', dtype=np.uint8, mode='r', shape=({filas_finales}, {columnas_finales_comprimidas}))")
print(f"# Descomprimir fila i")
print(f"row_bits = np.unpackbits(X[i, :], count={M})")
print(f"pattern = np.where(row_bits, 1, -1)  # {{0,1}} → {{-1,+1}}")
print(f"# Negar H2 (filas {N2}-{N-1})")
print(f"if i >= {N2}: pattern = -pattern")