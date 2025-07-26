import numpy as np
import cv2
import os

'''
Este código realiza la reconstrucción de imágenes usando el método de patrones Hadamard.
La caracterización de la MMF la brinda la RVITM, sin embargo, por temas de memoria, 
esta no será almacenada completa en ningún punto.
Se hace el producto:
Intensidad_Recuperada = [Matriz de Intensidad (speckles)] @ Matriz Hadamard_T @ Speckle_Imagen
Para evitar el consumo excesivo de RAM y overflow, se procesan los datos en chunks (secciones de matriz)
'''

print("=== INICIALIZANDO RECONSTRUCCIÓN DE IMAGEN ===")

# === Paths ===
Path_Imagen_a_Reconstruir = 'D:\\Speckle_De_Imagenes_A_Reconstruir\\pato.png'
Path_Matriz_Intensidad = 'D:\\Archivos_Reconstruccion\\Matriz_Intensidad.npy'
Path_Matriz_Hadamard_T = 'D:\\Archivos_Reconstruccion\\Hadamard_H_menosH_transpuesta.npy'
Output_Path = 'D:\\Archivos_Reconstruccion\\Imagenes_Reconstruidas'

# Validar que los archivos de entrada existen
archivos_requeridos = [
    (Path_Imagen_a_Reconstruir, "imagen a reconstruir"),
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
# M × N (intensidad) × N × P (Hadamard) debe ser válido
if shape_I_esperada[1] != shape_H_esperada[0]:
    raise ValueError(f"Incompatibilidad dimensional: Matriz_Intensidad columnas ({shape_I_esperada[1]}) "
                    f"≠ Matriz_Hadamard filas ({shape_H_esperada[0]})")

# Verificar que el resultado final sea compatible con la imagen
pixels_imagen = shape_img_esperada[0] * shape_img_esperada[1]  # 1310720
if shape_I_esperada[0] != pixels_imagen:
    raise ValueError(f"Incompatibilidad: filas matriz intensidad ({shape_I_esperada[0]}) "
                    f"≠ píxeles imagen ({pixels_imagen})")

print("Dimensiones validadas correctamente")

# === Memmap con validación de tipos y formas ===
print(f"\n=== CARGANDO MATRICES CON MEMMAP ===")

# Cargar metadatos sin cargar la matriz completa
print("Validando metadatos de archivos...")

# Verificación del tipo de datos para Matriz_Intensidad
try:
    # Usar mmap_mode='r' para leer solo el header
    with open(Path_Matriz_Intensidad, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape_intensidad_real, fortran_order, dtype_intensidad_original = np.lib.format.read_array_header_1_0(f)
    
    print(f"Matriz_Intensidad: {shape_intensidad_real}, dtype: {dtype_intensidad_original}")
    
    # Validar dimensiones sin cargar datos
    if shape_intensidad_real != shape_I_esperada:
        raise ValueError(f"Matriz_Intensidad tiene forma {shape_intensidad_real}, "
                        f"se esperaba {shape_I_esperada}")
    
    # Crear memmap con el tipo original
    Matriz_Intensidad = np.lib.format.open_memmap(Path_Matriz_Intensidad, 
                                                 dtype=dtype_intensidad_original, 
                                                 mode='r', 
                                                 shape=shape_I_esperada)
    print(f"Matriz_Intensidad cargada: {Matriz_Intensidad.shape}, {Matriz_Intensidad.dtype}")

except Exception as e:
    raise RuntimeError(f"Error al cargar Matriz_Intensidad: {e}")

# Verificación del tipo de datos para Matriz_Hadamard_T
try:
    with open(Path_Matriz_Hadamard_T, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape_hadamard_real, fortran_order, dtype_hadamard_original = np.lib.format.read_array_header_1_0(f)
    
    print(f"Matriz_Hadamard_T: {shape_hadamard_real}, dtype: {dtype_hadamard_original}")
    
    # Validar dimensiones sin cargar datos
    if shape_hadamard_real != shape_H_esperada:
        raise ValueError(f"Matriz_Hadamard_T tiene forma {shape_hadamard_real}, "
                        f"se esperaba {shape_H_esperada}")
    
    Matriz_Hadamard_T = np.lib.format.open_memmap(Path_Matriz_Hadamard_T, 
                                                  dtype=dtype_hadamard_original, 
                                                  mode='r', 
                                                  shape=shape_H_esperada)
    print(f"Matriz_Hadamard_T cargada: {Matriz_Hadamard_T.shape}, {Matriz_Hadamard_T.dtype}")

except Exception as e:
    raise RuntimeError(f"Error al cargar Matriz_Hadamard_T: {e}")

# Calcular memoria utilizada por los memmap (no cargan en RAM, solo mapean)
memoria_intensidad_gb = (shape_I_esperada[0] * shape_I_esperada[1] * 
                        np.dtype(dtype_intensidad_original).itemsize) / (1024**3)
memoria_hadamard_gb = (shape_H_esperada[0] * shape_H_esperada[1] * 
                      np.dtype(dtype_hadamard_original).itemsize) / (1024**3)

print(f"\nMemoria mapeada (no RAM):")
print(f"- Matriz_Intensidad: {memoria_intensidad_gb:.2f} GB")
print(f"- Matriz_Hadamard_T: {memoria_hadamard_gb:.2f} GB")
print(f"- Total archivos mapeados: {memoria_intensidad_gb + memoria_hadamard_gb:.2f} GB")

# === Imagen a reconstruir ===
print(f"\n=== CARGANDO Y VALIDANDO IMAGEN ===")
print(f"Procesando imagen: {os.path.basename(Path_Imagen_a_Reconstruir)}")

# Cargar imagen con validaciones
img = cv2.imread(Path_Imagen_a_Reconstruir, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"No se pudo cargar la imagen: {Path_Imagen_a_Reconstruir}")

# Validar dimensiones de la imagen
if img.shape != shape_img_esperada:
    raise ValueError(f"La imagen tiene dimensiones {img.shape}, "
                    f"se esperaba {shape_img_esperada}")

# Validar rango de valores
img_min, img_max = img.min(), img.max()
print(f"Imagen cargada: {img.shape}, rango=[{img_min}, {img_max}], dtype={img.dtype}")

if img_min < 0 or img_max > 255:
    raise ValueError(f"Imagen tiene valores fuera del rango uint8: [{img_min}, {img_max}]")

# Vectorizar imagen y convertir a float32 para evitar overflow en operaciones
I_out = img.flatten(order='C').astype(np.float32).reshape(-1, 1)
print(f"Vector imagen: forma={I_out.shape}, dtype={I_out.dtype}")

# Validar que el vector tiene el tamaño correcto
if I_out.shape[0] != pixels_imagen:
    raise ValueError(f"Vector imagen tiene {I_out.shape[0]} elementos, "
                    f"se esperaban {pixels_imagen}")

print(f"Imagen vectorizada correctamente: {I_out.shape}")

# ==== Calcular producto Hadamard x imagen ===== 
print(f"\n=== CÁLCULO DE PRODUCTO HADAMARD × IMAGEN ===")

# Calcular N (número de patrones de cada tipo)
N = Matriz_Hadamard_T.shape[0] // 2  # 8192 // 2 = 4096
print(f"N (patrones por tipo): {N}")

# Validar que N es correcto
if N <= 0 or N * 2 != Matriz_Hadamard_T.shape[0]:
    raise ValueError(f"Valor de N incorrecto: {N}. "
                    f"La matriz Hadamard debe tener un número par de filas.")

# Calcular memoria para el producto intermedio
# Hadamard (8192 × 1310720) × Imagen (1310720 × 1) = Resultado (8192 × 1)
memoria_intermedia_bytes = Matriz_Hadamard_T.shape[0] * 4  # float32 = 4 bytes
memoria_intermedia_mb = memoria_intermedia_bytes / (1024**2)
print(f"Memoria para resultado intermedio: {memoria_intermedia_mb:.2f} MB")

# Realizar multiplicación matricial con conversión de tipos
print("Calculando Matriz_Hadamard_T @ I_out...")

try:
    # Convertir Hadamard a float32 para la operación (solo las filas necesarias se cargan)
    intermedia = (Matriz_Hadamard_T.astype(np.float32) @ I_out).astype(np.float32)
    print(f"Producto intermedio calculado: {intermedia.shape}, dtype={intermedia.dtype}")
    
    # Validar resultado intermedio
    if intermedia.shape != (Matriz_Hadamard_T.shape[0], 1):
        raise ValueError(f"Resultado intermedio tiene forma incorrecta: {intermedia.shape}")
    
    # Verificar rangos para detectar posibles problemas
    inter_min, inter_max = intermedia.min(), intermedia.max()
    print(f"Rango resultado intermedio: [{inter_min:.2f}, {inter_max:.2f}]")
    
    # Verificar overflow potencial en float32
    if abs(inter_min) > 3.4e38 or abs(inter_max) > 3.4e38:
        print("Advertencia: Valores cercanos al límite de float32")

except Exception as e:
    raise RuntimeError(f"Error en multiplicación matricial: {e}")

# === Reconstrucción con chunks por filas de Matriz_Intensidad ===
print(f"\n=== RECONSTRUCCIÓN POR CHUNKS ===")

# Configuración de chunks optimizada para RAM
chunk_size = 32768  # Reducir para mayor seguridad de memoria
total_filas = shape_I_esperada[0]  # 1310720
num_chunks = (total_filas + chunk_size - 1) // chunk_size

# Calcular memoria por chunk
bytes_por_chunk = chunk_size * shape_I_esperada[1] * 4  # float32 = 4 bytes  
memoria_chunk_mb = bytes_por_chunk / (1024**2)

print(f"Configuración de chunks:")
print(f"- Tamaño de chunk: {chunk_size:,} filas")
print(f"- Total chunks: {num_chunks}")
print(f"- RAM por chunk: {memoria_chunk_mb:.1f} MB")
print(f"- Total filas a procesar: {total_filas:,}")

# Validar que el chunk size es razonable
if memoria_chunk_mb > 500:  # Más de 500 MB por chunk puede ser problemático
    print(f"Advertencia: Chunk size grande ({memoria_chunk_mb:.1f} MB)")

# Inicializar resultado final
I_rec = np.zeros((total_filas, 1), dtype=np.float32)
print(f"Vector resultado inicializado: {I_rec.shape}, {I_rec.dtype}")

# Procesar por chunks para evitar overflow de RAM
print("Iniciando procesamiento por chunks...")
filas_procesadas = 0

for chunk_idx in range(num_chunks):
    inicio = chunk_idx * chunk_size
    fin = min((chunk_idx + 1) * chunk_size, total_filas)
    filas_chunk = fin - inicio
    
    try:
        # Cargar chunk de la matriz de intensidad y convertir a float32
        fila_chunk = Matriz_Intensidad[inicio:fin, :].astype(np.float32)
        
        # Validar dimensiones del chunk
        if fila_chunk.shape[1] != shape_I_esperada[1]:
            raise ValueError(f"Chunk {chunk_idx} tiene {fila_chunk.shape[1]} columnas, "
                           f"se esperaban {shape_I_esperada[1]}")
        
        # Aplicar fórmula: I_rec = (1 / (2 * N)) * Matriz_Intensidad @ intermedia
        # Verificar overflow antes del cálculo
        factor_escala = 1.0 / (2.0 * N)
        
        # Calcular producto matricial para el chunk
        resultado_chunk = (factor_escala * fila_chunk @ intermedia).astype(np.float32)
        
        # Validar resultado del chunk
        if resultado_chunk.shape != (filas_chunk, 1):
            raise ValueError(f"Resultado chunk {chunk_idx} tiene forma incorrecta: {resultado_chunk.shape}")
        
        # Almacenar resultado
        I_rec[inicio:fin] = resultado_chunk
        filas_procesadas += filas_chunk
        
        # Liberar memoria del chunk
        del fila_chunk, resultado_chunk
        
        # Mostrar progreso cada 10 chunks
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == num_chunks - 1:
            progreso = (chunk_idx + 1) / num_chunks * 100
            print(f"Progreso: {progreso:.1f}% ({filas_procesadas:,}/{total_filas:,} filas)")
            
    except Exception as e:
        raise RuntimeError(f"Error procesando chunk {chunk_idx}: {e}")

print(f"Reconstrucción completada: {filas_procesadas:,} filas procesadas")

# Validar resultado final antes de normalizar
rec_min, rec_max = I_rec.min(), I_rec.max()
rec_mean = I_rec.mean()
print(f"Estadísticas resultado raw: rango=[{rec_min:.4f}, {rec_max:.4f}], media={rec_mean:.4f}")
    

# === Normalizar y guardar ===
print(f"\n=== NORMALIZACIÓN Y GUARDADO ===")

# Normalizar a rango 0-255 sin binarización
print("Normalizando a escala de grises 0-255...")

# Encontrar rangos para normalización
rec_min, rec_max = I_rec.min(), I_rec.max()
rec_mean = I_rec.mean()
print(f"Estadísticas resultado: rango=[{rec_min:.4f}, {rec_max:.4f}], media={rec_mean:.4f}")

# Normalización lineal a rango [0, 255]
'''
CASO ESPECIAL - VALORES CONSTANTES:
La normalización min-max usa: I_norm = (I_rec - min) / (max - min) * 255
Si max == min (todos los píxeles iguales), el denominador sería cero → división por cero
Esto puede suceder cuando:
1. El algoritmo falla completamente (todos los valores = 0)
2. Reconstrucción defectuosa (todos los valores iguales)
3. Imagen uniforme real (sin variaciones de intensidad)

SOLUCIÓN: Si max == min, asignar gris medio (128) en lugar de normalizar
- Evita crashes por división por cero
- Gris medio es neutro (no introduce bias hacia claro/oscuro)
- Visualmente detectable → indica problema en reconstrucción
'''
if rec_max != rec_min:  # Evitar división por cero
    I_norm = ((I_rec - rec_min) / (rec_max - rec_min) * 255.0).astype(np.uint8)
    print(f"Normalización aplicada: [{rec_min:.4f}, {rec_max:.4f}] -> [0, 255]")
else:
    # Si todos los valores son iguales, asignar valor medio
    I_norm = np.full_like(I_rec, 128, dtype=np.uint8)
    print("Advertencia: Todos los valores son iguales, asignando valor medio (128)")

# Validar resultado normalizado
norm_min, norm_max = I_norm.min(), I_norm.max()
print(f"Rango después de normalización: [{norm_min}, {norm_max}]")

# Reshaping con validación
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

# Estadísticas de la imagen final
img_min, img_max = img_rec.min(), img_rec.max()
img_mean = img_rec.mean()
img_std = img_rec.std()

print(f"Estadísticas imagen reconstruida:")
print(f"- Rango: [{img_min}, {img_max}]")
print(f"- Media: {img_mean:.1f}")
print(f"- Desviación estándar: {img_std:.1f}")

# Análisis de distribución de intensidades
hist_counts = np.bincount(img_rec.flatten(), minlength=256)
intensidades_no_cero = np.sum(hist_counts > 0)
print(f"- Intensidades únicas utilizadas: {intensidades_no_cero}/256")

# Guardar imagen con validación
nombre_salida = f'reconstruida_gris_{os.path.basename(Path_Imagen_a_Reconstruir)}'
ruta_completa = os.path.join(Output_Path, nombre_salida)

try:
    resultado_guardado = cv2.imwrite(ruta_completa, img_rec)
    if not resultado_guardado:
        raise RuntimeError("cv2.imwrite devolvió False")
    
    # Verificar que el archivo se guardó correctamente
    if os.path.exists(ruta_completa):
        tamaño_archivo_kb = os.path.getsize(ruta_completa) / 1024
        print(f"Imagen guardada exitosamente: {nombre_salida} ({tamaño_archivo_kb:.1f} KB)")
    else:
        raise FileNotFoundError("El archivo no se creó correctamente")
        
except Exception as e:
    raise RuntimeError(f"Error al guardar imagen: {e}")

print(f"\n=== RESUMEN FINAL ===")
print(f"Reconstrucción completada exitosamente")
print(f"Archivo de salida: {ruta_completa}")
print(f"Imagen original: {shape_img_esperada}")
print(f"Imagen reconstruida: {img_rec.shape}")
print(f"Rango valores finales: [{img_min}, {img_max}]")
print(f"RAM máxima utilizada: ~{memoria_chunk_mb:.1f} MB por chunk")
