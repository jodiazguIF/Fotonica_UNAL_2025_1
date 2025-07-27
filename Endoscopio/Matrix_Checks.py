import numpy as np
import os

# Paths to your data
path_speckles_H1 = '/media/manuel/Windows/Archivos_Reconstruccion/speckles_H1_vectorizados.npy'
path_speckles_H2 = '/media/manuel/Windows/Archivos_Reconstruccion/speckles_H2_vectorizados.npy'
path_H = '/media/manuel/Windows/Archivos_Reconstruccion/Hadamard_H_menosH_transpuesta.dat'  # Nueva matriz sin compresión

# Check if files exist
for path in [path_speckles_H1, path_speckles_H2, path_H]:
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        exit()

# Memory-map the .npy files so we don't load them fully into RAM
H1 = np.load(path_speckles_H1, mmap_mode='r')
H2 = np.load(path_speckles_H2, mmap_mode='r')

print("\nInspeccionando matrices de speckles...")

# Parámetros DMD para análisis espacial correcto
DMD_size = (1280, 1024)  # (ancho, alto)
ancho, alto = DMD_size

for name, arr in [('Speckles H1', H1), ('Speckles H2', H2)]:
    print(f"\n{name}:")
    print(f"  dtype = {arr.dtype}")
    print(f"  shape = {arr.shape} (pixels × patrones)")
    
    # Seleccionar patrones representativos para análisis espacial
    patrones_test = [0, arr.shape[1]//4, arr.shape[1]//2, 3*arr.shape[1]//4, arr.shape[1]-1]
    nombres_patrones = ['Primer', 'Cuarto', 'Medio', '3/4', 'Último']
    
    print(f"  Análisis espacial por patrones:")
    
    for idx, nombre in zip(patrones_test, nombres_patrones):
        if idx < arr.shape[1]:
            # Reconstruir patrón como imagen 2D
            patron_vec = arr[:, idx]  # Columna = patrón vectorizado
            img2d = patron_vec.reshape((alto, ancho), order='C')
            
            # Muestrear diferentes regiones de la imagen
            # Región 1: Esquina superior izquierda (padding potencial)
            patch_corner = img2d[:50, :50]
            
            # Región 2: Centro de la imagen (región activa probable)
            center_y, center_x = alto//2, ancho//2
            patch_center = img2d[center_y-25:center_y+25, center_x-25:center_x+25]
            
            # Región 3: Esquina inferior derecha
            patch_bottom_right = img2d[-50:, -50:]
            
            print(f"    {nombre} patrón (idx={idx}):")
            print(f"      Esquina sup-izq (50×50): min={patch_corner.min()}, max={patch_corner.max()}")
            print(f"      Centro (50×50):          min={patch_center.min()}, max={patch_center.max()}")
            print(f"      Esquina inf-der (50×50): min={patch_bottom_right.min()}, max={patch_bottom_right.max()}")
    
    # Estadísticas globales simplificadas (sin muestreo masivo)
    print(f"  Análisis estadístico rápido:")
    
    # Muestrear solo unos pocos patrones completos para estadísticas
    sample_patterns = [0, arr.shape[1]//2, arr.shape[1]-1]  # Solo 3 patrones
    all_values = []
    
    for pattern_idx in sample_patterns:
        if pattern_idx < arr.shape[1]:
            # Tomar solo una submuestra pequeña de cada patrón
            pattern_data = arr[::100, pattern_idx]  # Cada 100 pixels
            all_values.extend(pattern_data.tolist())
    
    if all_values:
        all_values = np.array(all_values)
        unique_vals = np.unique(all_values)
        print(f"    Valores únicos (3 patrones muestreados): {unique_vals}")
        print(f"    Rango: {all_values.min()} - {all_values.max()}")
        print(f"    Media: {all_values.mean():.3f}, Std: {all_values.std():.3f}")
        
        # Histograma solo si hay pocos valores únicos
        if len(unique_vals) <= 15:
            from collections import Counter
            counts = Counter(all_values)
            total = len(all_values)
            print(f"    Distribución: {dict((k, f'{v} ({v/total*100:.1f}%)') for k, v in sorted(counts.items()))}")
    else:
        print("    No se pudieron obtener muestras")
    
    print()

# Prepare memmap for [H, -H]ᵀ without full load
M = H1.shape[0]         # pixels per pattern
N2 = H1.shape[1]        # patterns per half
N  = 2 * N2             # total rows in H matrix

print(f"\n🔍 Verificación de compatibilidad dimensional:")
print(f"Speckles H1: {H1.shape[0]} pixels × {H1.shape[1]} patrones")
print(f"Speckles H2: {H2.shape[0]} pixels × {H2.shape[1]} patrones")
print(f"Esperado Hadamard: {N} filas × {M} columnas")

# Verificar consistencia
if H1.shape != H2.shape:
    print("WARNING: H1 y H2 tienen dimensiones diferentes")
else:
    print("1 y H2 tienen dimensiones consistentes")

if H1.shape[1] != N2:
    print(f"WARNING: Número de patrones inconsistente: {H1.shape[1]} vs {N2} esperado")
else:
    print("Número de patrones consistente con Hadamard")

if H1.shape[0] != M:
    print(f"WARNING: Número de pixels inconsistente: {H1.shape[0]} vs {M} esperado") 
else:
    print("Número de pixels consistente con DMD")

# Check actual file size and calculate correct dimensions
file_size = os.path.getsize(path_H)
expected_dtype = np.int8  # Expected dtype for Hadamard matrix
dtype_size = np.dtype(expected_dtype).itemsize  # Generic byte size calculation
expected_size = N * M * dtype_size
actual_elements = file_size // dtype_size

print(f"\nArchivo Hadamard (sin compresión): {file_size} bytes ({file_size/(1024**3):.2f} GB)")
print(f"Tipo esperado: {expected_dtype} ({dtype_size} bytes/elemento)")
print(f"Dimensiones esperadas: {N} x {M} = {N*M} elementos")
print(f"Tamaño esperado: {expected_size} bytes")
print(f"Elementos calculados: {actual_elements}")

if actual_elements != N * M:
    print("WARNING: Las dimensiones no coinciden. Calculando dimensiones reales...")
    # Try to find correct dimensions
    if actual_elements % M == 0:
        N_real = actual_elements // M
        print(f"Usando dimensiones corregidas: {N_real} x {M}")
        X = np.memmap(path_H, dtype=expected_dtype, mode='r', shape=(N_real, M))
    else:
        print("ERROR: No se pueden determinar dimensiones válidas")
        print(f"  Archivo: {actual_elements} elementos")
        print(f"  No es divisible por M={M}")
        exit()
else:
    print("Dimensiones correctas!")
    X = np.memmap(path_H, dtype=expected_dtype, mode='r', shape=(N, M))

print("\nMatriz [H, -H]ᵀ (sin compresión):")
print(f"  dtype = {X.dtype}")
print(f"  shape = {X.shape}")

# Sample first 100 rows and first 100 columns (zona de padding)
block = X[:100, :100]
print(f"  sample padding min = {block.min()}, sample padding max = {block.max()}")

# Muestreo más representativo: región central donde están los patrones activos
print(f"\n🔍 Muestreo representativo en región central activa:")

# Parámetros de escalado (calculados igual que en generación)
DMD_size = (1280, 1024)  # (ancho, alto)
ancho, alto = DMD_size
pattern_size = 64
scale = min(DMD_size) // pattern_size  # 16x escalado
scaled_size = pattern_size * scale  # 1024 pixels escalados
offset_x = (ancho - scaled_size) // 2  # 128 offset horizontal
offset_y = (alto - scaled_size) // 2   # 0 offset vertical

print(f"Región activa: offset_x={offset_x}, offset_y={offset_y}, tamaño={scaled_size}×{scaled_size}")

# Filas representativas para inspeccionar
filas_test = [0, N2//4, N2//2, 3*N2//4, N2, N2 + N2//4, N2 + N2//2, N-1]
nombres_filas = ['H1[0]', 'H1[1K]', 'H1[2K]', 'H1[3K]', 'H2[0]', 'H2[1K]', 'H2[2K]', 'H2[4K]']

for i, (fila, nombre) in enumerate(zip(filas_test, nombres_filas)):
    if fila < X.shape[0]:
        # Reconstruir como imagen 2D
        img2d = X[fila].reshape((alto, ancho), order='C')
        
        # Muestrear parche central activo (100×100 dentro de la región escalada)
        patch_size = 100
        start_x = offset_x + scaled_size//2 - patch_size//2
        start_y = offset_y + scaled_size//2 - patch_size//2
        patch = img2d[start_y:start_y+patch_size, start_x:start_x+patch_size]
        
        print(f"{nombre:8} parche central: min={patch.min():2d}, max={patch.max():2d} (valores únicos: {len(np.unique(patch))})")
        
        # Verificar que hay tanto +1 como -1 (o sus equivalentes)
        unique_vals = np.unique(patch)
        if len(unique_vals) >= 2 and patch.min() < 0 and patch.max() > 0:
            status = "✅"
        elif len(unique_vals) == 1 and (patch.min() == -1 or patch.max() == 1):
            status = "⚠️ "  # Solo un valor, pero válido
        else:
            status = "❌"
        print(f"        {status} Valores únicos en parche: {unique_vals}")

# Verificar H1 vs H2 en región central activa
print(f"\n🔍 Verificando relación H2 = -H1 en región central...")
if X.shape[0] >= 2 * N2:
    # Comparar patrones usando región central
    h1_img = X[0].reshape((alto, ancho), order='C')
    h2_img = X[N2].reshape((alto, ancho), order='C')
    
    # Extraer parche central de ambos
    patch_size = 200  # Parche más grande para verificación
    start_x = offset_x + scaled_size//2 - patch_size//2
    start_y = offset_y + scaled_size//2 - patch_size//2
    
    h1_patch = h1_img[start_y:start_y+patch_size, start_x:start_x+patch_size]
    h2_patch = h2_img[start_y:start_y+patch_size, start_x:start_x+patch_size]
    
    if np.array_equal(h2_patch, -h1_patch):
        print("H2 = -H1 verificado en región central (200×200)")
        print(f"H1 patch: min={h1_patch.min()}, max={h1_patch.max()}")
        print(f"H2 patch: min={h2_patch.min()}, max={h2_patch.max()}")
    else:
        print("WARNING: H2 ≠ -H1 en región central")
        print(f"H1 patch sample: {h1_patch[50:52, 50:52].flatten()}")
        print(f"H2 patch sample: {h2_patch[50:52, 50:52].flatten()}")
        print(f"-H1 patch sample: {(-h1_patch[50:52, 50:52]).flatten()}")
    
    # Verificar múltiples pares H1/H2 con parches 2D
    print(f"\nVerificando múltiples pares H1/H2...")
    indices_test = [0, N2//4, N2//2, 3*N2//4, N2-1]  # Índices representativos
    pares_correctos = 0
    
    for idx in indices_test:
        if idx < N2:  # Verificar que el índice sea válido
            # Reconstruir ambas imágenes como 2D
            img1 = X[idx].reshape((alto, ancho), order='C')
            img2 = X[N2 + idx].reshape((alto, ancho), order='C')
            
            # Extraer parches centrales
            patch_test_size = 50  # Parche más pequeño para test rápido
            start_x_test = offset_x + scaled_size//2 - patch_test_size//2
            start_y_test = offset_y + scaled_size//2 - patch_test_size//2
            
            p1 = img1[start_y_test:start_y_test+patch_test_size, start_x_test:start_x_test+patch_test_size]
            p2 = img2[start_y_test:start_y_test+patch_test_size, start_x_test:start_x_test+patch_test_size]
            
            if np.array_equal(p2, -p1):
                pares_correctos += 1
                print(f"Par H1[{idx}]/H2[{idx}]: correcto")
            else:
                print(f"Par H1[{idx}]/H2[{idx}]: FALLO")
    
    print(f"{pares_correctos}/{len(indices_test)} pares H1/H2 verificados correctamente")
    
    # Validación adicional con índices aleatorios
    print(f"\nValidación adicional con índices aleatorios...")
    np.random.seed(42)  # Para reproducibilidad
    indices_random = np.random.choice(N2, size=5, replace=False)
    pares_random_correctos = 0
    
    for idx in indices_random:
        img1 = X[idx].reshape((alto, ancho), order='C')
        img2 = X[N2 + idx].reshape((alto, ancho), order='C')
        
        # Parche central más pequeño para test rápido
        p1 = img1[start_y_test:start_y_test+patch_test_size, start_x_test:start_x_test+patch_test_size]
        p2 = img2[start_y_test:start_y_test+patch_test_size, start_x_test:start_x_test+patch_test_size]
        
        if np.array_equal(p2, -p1):
            pares_random_correctos += 1
    
    print(f"{pares_random_correctos}/5 pares aleatorios verificados correctamente")
    print(f"   Índices aleatorios testados: {indices_random}")
    
else:
    print("No hay suficientes filas para verificar H2")

print("\nVerificación completada.")
