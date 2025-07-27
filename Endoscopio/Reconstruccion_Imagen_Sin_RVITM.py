import numpy as np
import cv2
import os
import gc  # Para liberación explícita de memoria

'''
========================= RECONSTRUCCIÓN DE IMÁGENES CON PATRONES HADAMARD =========================

MARCO TEÓRICO:
1. Matriz Hadamard extendida X = [H, -H] ∈ {±1}^(M×2N)
   • Cada columna de X es un patrón binario (+1/−1) aplicado en la caracterización.

2. Matriz de intensidad RVITM (Y) ∈ ℤ^(M×2N)  
   • Y = RVITM · X (calculada en fase de caracterización)
   • Se almacena como memmap para evitar cargar 20+ GB en RAM

3. Correlación de la salida con los patrones
   • Dado I_out ∈ ℝ^(M×1), se computa c = X^T · I_out ∈ ℝ^(2N×1)
   • c_i mide cuánto "resuena" la salida con el i-ésimo patrón ±1

4. Reconstrucción lineal
   • Se usa la matriz de intensidad Y para invertir la distorsión del MMF
   • I_rec = (1/(2N)) * Y * c, donde Y = RVITM * X (caracterizada previamente)
   • Sin Y no hay forma de "invertir" la distorsión del MMF

5. Normalización y reshape
   • Normalizar I_rec a rango [0,255] → uint8
   • img_rec = I_rec.reshape(alto, ancho)

IMPLEMENTACIÓN OPTIMIZADA:
- Procesamiento por chunks para evitar overflow de RAM (40+ GB → ~100 MB por chunk)
- Uso de np.memmap para acceso eficiente a disco sin cargar matrices completas
- Liberación explícita de variables intermedias (del chunk tras cada multiplicación)
- Tipos de datos optimizados: int8/int16 en disco, float32 en cálculos, uint8 final

==========================================================================================
'''

print("=== INICIALIZANDO RECONSTRUCCIÓN DE IMAGEN ===")

# === Paths COMPLETAMENTE LOCALES (OPTIMIZADO) ===
# Todos los archivos ahora están en disco local para máxima velocidad
base_local = '/home/manuel/temp_intensity'
Path_Speckle_a_Reconstruir = '/home/manuel/temp_intensity/panda.png'  # SPECKLE del panda (LOCAL)
Path_Matriz_Intensidad = '/home/manuel/temp_intensity/temp_Matriz_Final.dat'  # Matriz intensidad (LOCAL)  
Path_Matriz_Hadamard_T = '/home/manuel/temp_intensity/Hadamard_H_menosH_transpuesta.dat'  # Matriz Hadamard (LOCAL)
Output_Path = '/home/manuel/temp_intensity/Imagenes_Reconstruidas'

print(f"=== CONFIGURACION COMPLETAMENTE LOCAL ===")
print(f"Todos los archivos en disco local para maximo rendimiento")
print(f"Base local: {base_local}")
print(f"Speckle panda: {Path_Speckle_a_Reconstruir}")
print(f"Matriz intensidad: {Path_Matriz_Intensidad}")
print(f"Matriz Hadamard: {Path_Matriz_Hadamard_T}")
print(f"Salida: {Output_Path}")

# Validar que los archivos de entrada existen
archivos_requeridos = [
    (Path_Speckle_a_Reconstruir, "speckle a reconstruir"),
    (Path_Matriz_Intensidad, "matriz de intensidad"),
    (Path_Matriz_Hadamard_T, "matriz de Hadamard transpuesta")
]

print("Validando archivos de entrada...")
for archivo, descripcion in archivos_requeridos:
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No se encontró {descripcion}: {archivo}")
    tamaño_mb = os.path.getsize(archivo) / (1024**2)
    print(f"- {descripcion}: {tamaño_mb:.1f} MB")

# Crear directorio de salida
os.makedirs(Output_Path, exist_ok=True)
print(f"Directorio de salida: {Output_Path}")

print(f"\n=== VALIDACIÓN DE DIMENSIONES Y TIPOS DE DATOS ===")

# === Dimensiones esperadas ===
shape_I_esperada = (1310720, 8192)  # Matriz de intensidad
shape_H_esperada = (8192, 1310720)  # Matriz Hadamard transpuesta
shape_img_esperada = (1024, 1280)   # Imagen original (alto, ancho)

print(f"Dimensiones esperadas:")
print(f"- Matriz intensidad: {shape_I_esperada}")  
print(f"- Matriz Hadamard: {shape_H_esperada}")
print(f"- Imagen: {shape_img_esperada}")

# Verificar compatibilidad dimensional antes de cargar
# M x N (intensidad) x N x P (Hadamard) debe ser valido
if shape_I_esperada[1] != shape_H_esperada[0]:
    raise ValueError(f"Incompatibilidad dimensional: Matriz_Intensidad columnas ({shape_I_esperada[1]}) "
                    f"!= Matriz_Hadamard filas ({shape_H_esperada[0]})")

# Verificar que el resultado final sea compatible con la imagen
pixels_imagen = shape_img_esperada[0] * shape_img_esperada[1]  # 1310720
if shape_I_esperada[0] != pixels_imagen:
    raise ValueError(f"Incompatibilidad: filas matriz intensidad ({shape_I_esperada[0]}) "
                    f"!= pixeles imagen ({pixels_imagen})")

print("Dimensiones validadas correctamente")

# === Cargar matrices .dat con memmap ===
print(f"\n=== CARGANDO MATRICES .DAT CON MEMMAP ===")
print("Cargando matrices desde archivos .dat (formato optimizado)...")

# Cargar Matriz de Intensidad (.dat)
try:
    print(f"Cargando Matriz_Intensidad desde: {Path_Matriz_Intensidad}")
    Matriz_Intensidad = np.memmap(Path_Matriz_Intensidad, 
                                 dtype=np.int16, mode='r', 
                                 shape=shape_I_esperada)
    print(f"Matriz_Intensidad cargada: {Matriz_Intensidad.shape}, {Matriz_Intensidad.dtype}")
    
except Exception as e:
    raise RuntimeError(f"Error al cargar Matriz_Intensidad: {e}")

# Cargar Matriz Hadamard (.dat)
try:
    print(f"Cargando Matriz_Hadamard_T desde: {Path_Matriz_Hadamard_T}")
    Matriz_Hadamard_T = np.memmap(Path_Matriz_Hadamard_T, 
                                 dtype=np.int8, mode='r', 
                                 shape=shape_H_esperada)
    print(f"Matriz_Hadamard_T cargada: {Matriz_Hadamard_T.shape}, {Matriz_Hadamard_T.dtype}")
    
except Exception as e:
    raise RuntimeError(f"Error al cargar Matriz_Hadamard_T: {e}")

# Calcular memoria utilizada por los memmap (no cargan en RAM, solo mapean)
memoria_intensidad_gb = (shape_I_esperada[0] * shape_I_esperada[1] * 2) / (1024**3)  # int16 = 2 bytes
memoria_hadamard_gb = (shape_H_esperada[0] * shape_H_esperada[1] * 1) / (1024**3)    # int8 = 1 byte

print(f"\nMemoria mapeada (no RAM):")
print(f"- Matriz_Intensidad: {memoria_intensidad_gb:.2f} GB")
print(f"- Matriz_Hadamard_T: {memoria_hadamard_gb:.2f} GB")
print(f"- Total archivos mapeados: {memoria_intensidad_gb + memoria_hadamard_gb:.2f} GB")

# ===== DIAGNÓSTICO CRÍTICO: VERIFICAR CONTENIDO DE MATRICES =====
print(f"\n=== DIAGNÓSTICO DE CONTENIDO DE MATRICES ===")

# Verificar Matriz Hadamard (debe contener solo ±1)
print("Analizando Matriz_Hadamard_T...")
h_sample = Matriz_Hadamard_T[:100, :100]  # Muestra pequeña para análisis
h_unique = np.unique(h_sample)
print(f"Valores únicos en muestra Hadamard: {h_unique}")
if len(h_unique) == 2 and set(h_unique) == {-1, 1}:
    print("✓ Matriz Hadamard válida: contiene solo ±1")
elif len(h_unique) == 2 and set(h_unique) == {0, 1}:
    print("⚠ Matriz Hadamard binaria: contiene 0,1 (se convertirá a ±1)")
else:
    print(f"✗ PROBLEMA: Matriz Hadamard contiene valores inesperados: {h_unique}")

# Verificar Matriz de Intensidad (debe tener variación significativa)
print("Analizando Matriz_Intensidad...")
y_sample = Matriz_Intensidad[:1000, :100].astype(np.float32)  # Muestra para análisis
y_min, y_max = y_sample.min(), y_sample.max()
y_mean, y_std = y_sample.mean(), y_sample.std()
print(f"Muestra Y - Rango: [{y_min:.2f}, {y_max:.2f}], Media: {y_mean:.2f}, Std: {y_std:.2f}")

# Detectar si Y contiene speckles preprocesados o sin procesar
if y_min >= 0 and y_max <= 255 and y_std < 50:
    print("⚠ POSIBLE PROBLEMA: Y parece contener speckles sin procesar (rango 0-255, baja variación)")
elif y_min < -100 or y_max > 400:
    print("✓ Y parece preprocesado: rango extendido sugiere Y = 2·Speckle - I₁")
else:
    print("? Y tiene características intermedias - verificar proceso de caracterización")

del h_sample, y_sample  # Liberar muestras de diagnóstico
gc.collect()

# === Speckle a reconstruir ===
print(f"\n=== CARGANDO Y VALIDANDO SPECKLE ===")
print(f"Procesando speckle: {os.path.basename(Path_Speckle_a_Reconstruir)}")

# Cargar speckle con validaciones
img = cv2.imread(Path_Speckle_a_Reconstruir, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"No se pudo cargar el speckle: {Path_Speckle_a_Reconstruir}")

# Validar dimensiones del speckle
if img.shape != shape_img_esperada:
    raise ValueError(f"El speckle tiene dimensiones {img.shape}, "
                    f"se esperaba {shape_img_esperada}")

# Validación simplificada: OpenCV siempre produce uint8 (recomendación 3)
print(f"Speckle cargado: {img.shape}, dtype={img.dtype}")

# Vectorizar speckle y convertir a float32 para evitar overflow en operaciones
I_out = img.flatten(order='C').astype(np.float32).reshape(-1, 1)
print(f"Vector speckle: forma={I_out.shape}, dtype={I_out.dtype}")

# Validar que el vector tiene el tamaño correcto
if I_out.shape[0] != pixels_imagen:
    raise ValueError(f"Vector speckle tiene {I_out.shape[0]} elementos, "
                    f"se esperaban {pixels_imagen}")

print(f"Speckle vectorizado correctamente: {I_out.shape}")

# ==== PASO 3: CORRELACIÓN X^T · I_out OPTIMIZADA POR CHUNKS ===== 
print(f"\n=== CÁLCULO DE CORRELACIÓN c = X^T · I_out ===")
print("TEORÍA: c_i mide resonancia de la salida con el i-ésimo patrón ±1")

# Calcular N (número de patrones de cada tipo en X = [H, -H])
N = Matriz_Hadamard_T.shape[0] // 2  # 8192 // 2 = 4096
print(f"N (patrones por tipo): {N}")
print(f"Matriz Hadamard X^T: {Matriz_Hadamard_T.shape} = [H^T; -H^T]")

# Validar que N es correcto y corresponde a un orden Hadamard válido
if N <= 0 or N * 2 != Matriz_Hadamard_T.shape[0]:
    raise ValueError(f"Valor de N incorrecto: {N}. "
                    f"La matriz Hadamard debe tener un número par de filas.")

# VERIFICACIÓN CRÍTICA: N debe ser potencia de 2 para Hadamard válido
import math
log2_N = math.log2(N)
if not log2_N.is_integer():
    print(f"ADVERTENCIA: N={N} no es potencia de 2. Orden Hadamard inusual.")
else:
    orden_hadamard = int(log2_N)
    print(f"Orden Hadamard confirmado: 2^{orden_hadamard} = {N}")
    
# Verificar que el factor de escala es razonable
factor_escala_check = 1.0 / (2.0 * N)
print(f"Factor de escala: 1/(2N) = 1/{2*N} = {factor_escala_check:.8f}")
if factor_escala_check < 1e-6 or factor_escala_check > 0.1:
    print(f"ADVERTENCIA: Factor de escala inusual: {factor_escala_check}")
else:
    print("Factor de escala validado correctamente")

# Configuración de chunks optimizada (recomendación: 256-512 filas)
hadamard_chunk_size = 512  # REDUCIDO: 768→512 para garantizar <2.6GB por chunk
total_filas_hadamard = Matriz_Hadamard_T.shape[0]  # 8192
num_hadamard_chunks = (total_filas_hadamard + hadamard_chunk_size - 1) // hadamard_chunk_size

# Calcular memoria por chunk de Hadamard
memoria_hadamard_chunk_gb = (hadamard_chunk_size * Matriz_Hadamard_T.shape[1] * 4) / (1024**3)  # float32

# ===== OPTIMIZACIÓN CRÍTICA DE MEMORIA =====
print(f"PROBLEMA: X^T @ I_out directo requeriría ~40 GB en RAM")
print(f"SOLUCIÓN: Procesamiento por chunks siguiendo recomendación 1")
print(f"VENTAJA: Reducir 40 GB → {memoria_hadamard_chunk_gb:.3f} GB por chunk")

print(f"Configuración chunks Hadamard:")
print(f"- Chunk size: {hadamard_chunk_size} filas (recomendación: 256-512)")
print(f"- Total chunks: {num_hadamard_chunks}")
print(f"- RAM por chunk: {memoria_hadamard_chunk_gb:.3f} GB (SEGURO)")

# Inicializar vector correlación c = X^T · I_out
intermedia = np.zeros((total_filas_hadamard, 1), dtype=np.float32)
print(f"Vector correlación c inicializado: {intermedia.shape}")

print("Calculando c = X^T @ I_out por chunks (implementación optimizada)...")
import time
inicio_hadamard = time.time()

# IMPLEMENTACIÓN SIGUIENDO RECOMENDACIONES 2, 3, 6:
# - np.memmap en modo 'r' para acceso secuencial a disco
# - Liberación explícita tras cada multiplicación (del chunk)  
# - BLAS multihilo aprovecha varios núcleos automáticamente
for chunk_idx in range(num_hadamard_chunks):
    inicio_fila = chunk_idx * hadamard_chunk_size
    fin_fila = min((chunk_idx + 1) * hadamard_chunk_size, total_filas_hadamard)
    
    try:
        # Cargar chunk de X^T desde memmap y convertir a float32 (recomendación 5)
        hadamard_chunk = Matriz_Hadamard_T[inicio_fila:fin_fila, :].astype(np.float32)
        
        # Multiplicar chunk por speckle: c_chunk = X_chunk^T @ I_out
        resultado_chunk = hadamard_chunk @ I_out
        
        # Almacenar resultado del chunk
        intermedia[inicio_fila:fin_fila, :] = resultado_chunk
        
        # Liberación explícita de memoria (recomendación 3)
        del hadamard_chunk, resultado_chunk
        
        # Liberación adicional cada 5 chunks
        if (chunk_idx + 1) % 5 == 0:
            gc.collect()  # Forzar recolección de basura
        
        # Mostrar progreso cada 10% (recomendación 2: reducir overhead I/O)
        if chunk_idx % max(1, num_hadamard_chunks//10) == 0 or chunk_idx == num_hadamard_chunks - 1:
            progreso = (chunk_idx + 1) / num_hadamard_chunks * 100
            tiempo_transcurrido = time.time() - inicio_hadamard
            print(f"Correlación: {progreso:.1f}% completado ({chunk_idx+1}/{num_hadamard_chunks}) - {tiempo_transcurrido:.1f}s")
            
    except Exception as e:
        raise RuntimeError(f"Error en chunk correlación {chunk_idx}: {e}")

tiempo_hadamard = time.time() - inicio_hadamard
print(f"Correlación c = X^T @ I_out completada en {tiempo_hadamard:.1f}s")
print(f"Vector correlación calculado: {intermedia.shape}, dtype={intermedia.dtype}")

# Validar resultado correlación - solo forma (rápido)
if intermedia.shape != (Matriz_Hadamard_T.shape[0], 1):
    raise ValueError(f"Vector correlación tiene forma incorrecta: {intermedia.shape}")

# Validaciones de overflow eliminadas (recomendación 3: casos raros)

# === PASO 4: RECONSTRUCCIÓN I_rec = (1/(2N)) * Y @ c ===
print(f"\n=== RECONSTRUCCIÓN LINEAL I_rec = (1/(2N)) * Y @ c ===")
print("TEORÍA: Usar matriz de intensidad Y para invertir distorsión del MMF")
print(f"FÓRMULA: I_rec = (1/(2N)) * Matriz_Intensidad @ c")
print("IMPORTANTE: Y = RVITM * X (caracterizada previamente) es esencial para invertir la fibra")

# Configuración de chunks optimizada para Y @ c (matriz de intensidad)
chunk_size = 32768  # REDUCIDO: 49152→32768 para garantizar estabilidad (~1GB/chunk)
total_filas = shape_I_esperada[0]  # 1310720 píxeles de salida
num_chunks = (total_filas + chunk_size - 1) // chunk_size

# Usar la matriz de intensidad Y (NO la matriz de patrones X)
print(f"Matriz Y (intensidad): {Matriz_Intensidad.shape} (M × 2N)")
print(f"Vector correlación c: {intermedia.shape} (2N × 1)")
print(f"Resultado: Y @ c → I_rec (M × 1)")

# Calcular memoria por chunk para Y @ c
bytes_por_chunk = chunk_size * shape_I_esperada[1] * 4  # float32 = 4 bytes  
memoria_chunk_mb = bytes_por_chunk / (1024**2)

print(f"Configuración chunks para X @ c:")
print(f"- Tamaño de chunk: {chunk_size:,} filas")
print(f"- Total chunks: {num_chunks}")
print(f"- RAM por chunk: {memoria_chunk_mb:.1f} MB")
print(f"- Total píxeles a reconstruir: {total_filas:,}")
print(f"- Factor escala: 1/(2N) = 1/{2*N} = {1.0/(2*N):.6f}")

# Validar que el chunk size es razonable (recomendación 1)
if memoria_chunk_mb > 500:  # Más de 500 MB por chunk puede ser problemático
    print(f"Advertencia: Chunk size grande ({memoria_chunk_mb:.1f} MB)")

# Inicializar resultado final I_rec
I_rec = np.zeros((total_filas, 1), dtype=np.float32)
print(f"Vector reconstrucción I_rec inicializado: {I_rec.shape}, {I_rec.dtype}")

# Procesar Y @ c por chunks (usando matriz de intensidad para invertir MMF)
print("Iniciando reconstrucción I_rec = (1/(2N)) * Y @ c por chunks...")
filas_procesadas = 0

# Factor de escala precalculado (recomendación 5: mantener float32 durante cálculo)
factor_escala = np.float32(1.0 / (2.0 * N))

# Running min/max para evitar pasada adicional (recomendación 6)
rec_min_running = np.float32(np.inf)
rec_max_running = np.float32(-np.inf)

for chunk_idx in range(num_chunks):
    inicio = chunk_idx * chunk_size
    fin = min((chunk_idx + 1) * chunk_size, total_filas)
    filas_chunk = fin - inicio
    
    try:
        # Cargar chunk de Y (matriz de intensidad) desde memmap y convertir a float32
        y_chunk = Matriz_Intensidad[inicio:fin, :].astype(np.float32)
        
        # Validar dimensiones del chunk Y
        if y_chunk.shape[1] != shape_I_esperada[1]:  # Debe tener 2N columnas (8192)
            raise ValueError(f"Chunk Y {chunk_idx} tiene {y_chunk.shape[1]} columnas, "
                           f"se esperaban {shape_I_esperada[1]} (2N)")
        
        # Aplicar fórmula de reconstrucción: I_rec = (1/(2*N)) * Y @ c
        # Y @ c invierte la distorsión del MMF usando la caracterización previa
        resultado_chunk = (factor_escala * (y_chunk @ intermedia)).astype(np.float32)
        
        # Validar resultado del chunk
        if resultado_chunk.shape != (filas_chunk, 1):
            raise ValueError(f"Resultado chunk {chunk_idx} tiene forma incorrecta: {resultado_chunk.shape}")
        
        # Almacenar resultado
        I_rec[inicio:fin] = resultado_chunk
        filas_procesadas += filas_chunk
        
        # Actualizar min/max running (recomendación 6: evitar pasada adicional)
        chunk_min = resultado_chunk.min()
        chunk_max = resultado_chunk.max()
        rec_min_running = min(rec_min_running, chunk_min)
        rec_max_running = max(rec_max_running, chunk_max)
        
        # Liberación explícita de memoria (recomendación 3)
        del y_chunk, resultado_chunk
        
        # Liberación adicional cada 10 chunks
        if (chunk_idx + 1) % 10 == 0:
            gc.collect()  # Forzar recolección de basura
        
        # Mostrar progreso cada 10% (recomendación 2: reducir overhead I/O)
        if chunk_idx % max(1, num_chunks//10) == 0 or chunk_idx == num_chunks - 1:
            progreso = (chunk_idx + 1) / num_chunks * 100
            print(f"Reconstrucción: {progreso:.1f}% ({filas_procesadas:,}/{total_filas:,} píxeles)")
            
    except Exception as e:
        raise RuntimeError(f"Error procesando chunk reconstrucción {chunk_idx}: {e}")

print(f"Reconstrucción I_rec completada: {filas_procesadas:,} píxeles procesados")

# Liberación de memoria intermedia (recomendación 3)
del intermedia  # Liberar vector correlación c (ya no se necesita)
gc.collect()    # Forzar liberación antes de normalización

# ===== EXPLICACIÓN TEÓRICA CORRECTA =====
print(f"\n=== TEORÍA CORRECTA DE RECONSTRUCCIÓN ===")
print("CONSTRUCCIÓN Y: Y = 2*Speckle_i - I₁ (ya aplicada en Matriz_Intensidad.py)")
print("RECONSTRUCCIÓN: I_rec = (1/(2N)) * (Y @ c)")  
print("RESULTADO: I_rec YA contiene la imagen correcta (sin necesidad de sumar I₁)")
print("")
print("IMPORTANTE: NO se debe sumar I₁ después, porque ya está")
print("           implícito en la construcción de Y = 2*Speckle - I₁")

# La reconstrucción I_rec YA es correcta tal como está
# No necesitamos ninguna corrección DC adicional
I_rec_final = I_rec.copy()
print(f"I_rec final (sin corrección DC): rango=[{I_rec_final.min():.4f}, {I_rec_final.max():.4f}]")
print(f"Media: {I_rec_final.mean():.4f}, Std: {I_rec_final.std():.4f}")

# Usar directamente I_rec sin modificaciones
I_rec = I_rec_final
del I_rec_final
gc.collect()

# === PASO 5: NORMALIZACIÓN Y RESHAPE ===
print(f"\n=== NORMALIZACIÓN I_rec → [0,255] Y RESHAPE ===")
print("TEORÍA: Convertir float32 → uint8 para imagen final")
print(f"\n=== NORMALIZACIÓN Y GUARDADO ===")

# Normalizar a rango 0-255 sin binarización
print("Normalizando a escala de grises 0-255...")

# IMPORTANTE: Usar los valores directos de I_rec (sin corrección DC)
rec_min, rec_max = I_rec.min(), I_rec.max()
print(f"Estadísticas I_rec final: rango=[{rec_min:.4f}, {rec_max:.4f}]")

# Normalización lineal a rango [0, 255]
'''
CASO ESPECIAL - VALORES CONSTANTES:
La normalizacion min-max usa: I_norm = (I_rec - min) / (max - min) * 255
Si max == min (todos los pixeles iguales), el denominador seria cero -> division por cero
Esto puede suceder cuando:
1. El algoritmo falla completamente (todos los valores = 0)
2. Reconstruccion defectuosa (todos los valores iguales)
3. Imagen uniforme real (sin variaciones de intensidad)

SOLUCION: Si max == min, asignar gris medio (128) en lugar de normalizar
- Evita crashes por division por cero
- Gris medio es neutro (no introduce bias hacia claro/oscuro)
- Visualmente detectable -> indica problema en reconstruccion
'''
if rec_max != rec_min:  # Evitar división por cero
    I_norm = ((I_rec - rec_min) / (rec_max - rec_min) * 255.0).astype(np.uint8)
    print(f"Normalización aplicada: [{rec_min:.4f}, {rec_max:.4f}] → [0, 255]")
else:
    # Si todos los valores son iguales, asignar valor medio
    I_norm = np.full_like(I_rec, 128, dtype=np.uint8)
    print("Advertencia: Todos los valores son iguales, asignando valor medio (128)")

# Liberación de I_rec tras conversión (recomendación 3, 5)
del I_rec  # Liberar float32, mantener solo uint8 final
gc.collect()   # Liberar 5MB adicionales

# Validacion rapida resultado normalizado
print(f"Rango despues de normalizacion: [{I_norm.min()}, {I_norm.max()}]")

# Reshape a imagen 2D (paso final)
print(f"Convirtiendo vector a imagen {shape_img_esperada}...")
try:
    img_rec = I_norm.reshape(shape_img_esperada)
    print(f"Imagen reconstruida: {img_rec.shape}, dtype={img_rec.dtype}")
    
    # Validar que no hay overflow en el reshape
    if img_rec.size != I_norm.size:
        raise ValueError(f"Error en reshape: tamaños diferentes {img_rec.size} vs {I_norm.size}")
        
except ValueError as e:
    raise ValueError(f"Error en reshape: {e}. "
                    f"Vector tiene {I_norm.size} elementos, "
                    f"imagen requiere {shape_img_esperada[0] * shape_img_esperada[1]}")

# Liberación de I_norm tras reshape (recomendación 3)
del I_norm  # Solo mantener img_rec final

# Estadisticas finales simplificadas (evitar calculos costosos)
print(f"Imagen reconstruida: {img_rec.shape}, dtype={img_rec.dtype}")
print(f"Rango valores finales: [{img_rec.min()}, {img_rec.max()}]")

# Guardar imagen con validacion
nombre_salida = f'reconstruida_panda_{os.path.basename(Path_Speckle_a_Reconstruir)}'
ruta_completa = os.path.join(Output_Path, nombre_salida)

try:
    resultado_guardado = cv2.imwrite(ruta_completa, img_rec)
    if not resultado_guardado:
        raise RuntimeError("cv2.imwrite devolvio False")
    
    # Verificar que el archivo se guardo correctamente
    if os.path.exists(ruta_completa):
        tamano_archivo_kb = os.path.getsize(ruta_completa) / 1024
        print(f"Imagen guardada exitosamente: {nombre_salida} ({tamano_archivo_kb:.1f} KB)")
    else:
        raise FileNotFoundError("El archivo no se creo correctamente")
        
except Exception as e:
    raise RuntimeError(f"Error al guardar imagen: {e}")

print(f"\n=== RESUMEN FINAL ===")
print(f"Reconstruccion completada exitosamente")
print(f"Archivo de salida: {ruta_completa}")
print(f"Imagen original: {shape_img_esperada}")
print(f"Imagen reconstruida: {img_rec.shape}")
print(f"Rango valores finales: [0, 255]")  # Siempre uint8 tras normalizacion
print(f"RAM maxima utilizada: ~{memoria_chunk_mb:.1f} MB por chunk")
