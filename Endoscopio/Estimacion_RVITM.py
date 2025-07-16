import numpy as np

# Cargar matrices como float32 para ahorrar RAM
Y = np.load('/media/manuel/Windows/Matriz_Intensidad.npy').astype(np.float32)     # (131072, 8192)
X = np.load('/media/manuel/Windows/Hadamard_H_menosH.npy').astype(np.float32)     # (131072, 8192)

# Verifica formas
assert Y.shape[0] == X.shape[0], "Las matrices Y y X no tienen el mismo número de filas (pixeles)"
assert Y.shape[1] == X.shape[1], "Y y X deben tener el mismo número de columnas (patrones)"

# Número de patrones originales de Hadamard (N)
N = X.shape[1] // 2

# Estimar RVITM con float32
print("Estimando RVITM... esto puede tardar unos minutos.")
RVITM = (1 / (2 * N)) * (Y @ X.T)  # Resultado: (131072, 131072), tipo float32

# Guardar la matriz
np.save('/media/manuel/Windows/RVITM_estimada.npy', RVITM)

print("RVITM estimada y guardada como: /media/manuel/Windows/RVITM_estimada.npy")
