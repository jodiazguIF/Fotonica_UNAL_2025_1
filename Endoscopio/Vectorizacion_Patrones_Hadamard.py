import numpy as np
import os
import time
import shutil
from scipy.linalg import hadamard

'''
Vectorización de patrones de Hadamard proyectados en el DMD.
Genera matriz [H, -H]^T sin compresión, valores int8 directos ±1 para acceso rápido.
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

def make_hadamard_pattern_optimized(H_full, col_idx, pattern_size, DMD_size, scale, offset_x, offset_y, canvas_buffer, scale_idxs):
    """
    Genera patrón Hadamard individual escalado y centrado para DMD
    
    Args:
        H_full: Matriz Hadamard completa N²×N²
        col_idx: Índice de columna a extraer
        pattern_size: Tamaño base (64)
        DMD_size: Dimensiones DMD (ancho, alto)
        scale: Factor de escalado
        offset_x, offset_y: Offsets de centrado
        canvas_buffer: Buffer reutilizable
        scale_idxs: Índices precomputados para escalado rápido
    
    Returns:
        np.array: Patrón vectorizado de tamaño M
    """
    # Extraer columna de matriz Hadamard
    columna = H_full[:, col_idx]
    
    # Reshape a matriz cuadrada 64×64
    square = columna.reshape((pattern_size, pattern_size), order='C')
    
    # Escalado optimizado con indexación precomputada (10-20x más rápido)
    square_scaled = square[scale_idxs][:, scale_idxs]
    
    # Limpiar y reutilizar canvas DMD
    canvas_buffer.fill(0)
    canvas_buffer[offset_y:offset_y+square_scaled.shape[0], offset_x:offset_x+square_scaled.shape[1]] = square_scaled
    
    # Vectorizar con orden C (row-major)
    return canvas_buffer.ravel(order='C')

# Configurar paths de almacenamiento
temp_dir = '/home/manuel/temp_hadamard'
os.makedirs(temp_dir, exist_ok=True)
final_path_ntfs = '/media/manuel/Windows/Archivos_Reconstruccion/Hadamard_H_menosH_transpuesta.dat'
temp_final_path = os.path.join(temp_dir, 'X_T_final.dat')

print(f"Directorio temporal: {temp_dir}")
print(f"Archivo final: {final_path_ntfs}")
print(f"Tamaño final: {N * M / (1024**3):.2f} GB sin compresión")

# Limpiar archivos previos
archivos_limpiar = [
    os.path.join(temp_dir, 'X_T_final.dat'),
    '/media/manuel/Windows/Archivos_Reconstruccion/Hadamard_H_menosH_transpuesta.dat'
]

for archivo in archivos_limpiar:
    if os.path.exists(archivo):
        try:
            os.remove(archivo)
            print(f"Eliminado: {os.path.basename(archivo)}")
        except Exception as e:
            print(f"Error eliminando {os.path.basename(archivo)}: {e}")

# Crear matriz final directamente sin compresión
print(f"Creando matriz final {N} × {M} (sin compresión)...")
tiempo_inicio = time.time()

# Memmap raw con valores ±1
X = np.memmap(temp_final_path, dtype=np.int8, mode='w+', shape=(N, M))

print("\n=== GENERANDO MATRIZ [H, -H]^T OPTIMIZADA ===")

# Generar patrones optimizados evitando cálculos duplicados
print("Generando patrones con optimizaciones de CPU...")

# Buffer reutilizable y índices precomputados para escalado rápido
canvas_buffer = np.zeros(DMD_size[::-1], dtype=np.int8)  # (alto, ancho)
scale_idxs = np.repeat(np.arange(pattern_size), scale)  # Índices para escalado en C

print("Generando solo patrones H1 (filas 0-4095)...")
for i in range(N2):
    X[i, :] = make_hadamard_pattern_optimized(H_full, i, pattern_size, DMD_size, scale, offset_x, offset_y, canvas_buffer, scale_idxs)
    if (i + 1) % 500 == 0:
        progreso_h1 = (i + 1) / N2 * 100
        tiempo_parcial = time.time() - tiempo_inicio
        velocidad = (i + 1) / tiempo_parcial
        print(f"H1: {i+1}/{N2} ({progreso_h1:.1f}%) - {velocidad:.1f} pat/s")

print("Copiando H2 como -H1 (sin recálculo)...")
# Copia vectorizada eficiente: H2 = -H1
X[N2:N, :] = -X[0:N2, :]

print(f"H2 generado por copia vectorizada (evita {N2} cálculos duplicados)")

X.flush()
tiempo_construccion = time.time() - tiempo_inicio
print(f"Matriz final construida en {tiempo_construccion:.1f}s")

# Mover archivo final a destino
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
X = None

try:
    os.rmdir(temp_dir)
    print(f"Directorio temporal eliminado")
except OSError:
    print(f"Directorio temporal mantenido (no vacío)")

# Resumen final
tiempo_total = time.time() - tiempo_inicio
tamaño_final = N * M / (1024**3)  # Recalcular tamaño en GB

print(f"\n=== GENERACIÓN COMPLETADA (OPTIMIZADA) ===")
print(f"Archivo: {final_path_ntfs}")
print(f"Dimensiones: {N} × {M} (sin compresión)")
print(f"Tamaño: {tamaño_final:.1f} GB")
print(f"Tiempo total: {tiempo_total:.1f}s ({N/tiempo_total:.1f} pat/s)")
print(f"H1: {N2} patrones generados, H2: {N2} patrones copiados")
print(f"Escalado: {scale}x con indexación precomputada")
print(f"Optimizaciones: escalado ~15x más rápido + copia vectorizada H2")
print(f"Centrado: ({offset_x}, {offset_y})")

print(f"\n=== USO DIRECTO ===")
print(f"# Cargar matriz")
print(f"X = np.memmap('{final_path_ntfs}', dtype=np.int8, mode='r', shape=({N}, {M}))")
print(f"# Acceso directo sin descompresión")
print(f"patron_i = X[i, :]  # Valores directos ±1")
print(f"# H1: filas 0-{N2-1}, H2: filas {N2}-{N-1}")