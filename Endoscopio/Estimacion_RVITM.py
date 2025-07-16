import numpy as np

# Cargar matrices
Y = np.load('/media/manuel/Windows/Matriz_Intensidad.npy')         # (131072, 8192)
X = np.load('/media/manuel/Windows/Hadamard_H_menosH.npy')        # (131072, 8192)

# Verifica formas
assert Y.shape[0] == X.shape[0], "Las matrices Y y X no tienen el mismo número de filas (pixeles)"
assert Y.shape[1] == X.shape[1], "Y y X deben tener el mismo número de columnas (patrones)"

# Número de patrones originales de Hadamard (N)
N = X.shape[1] // 2

# Estimar RVITM (usa la propiedad ortogonal de Hadamard)
RVITM = (1 / (2 * N)) * (Y @ X.T)  # resultado: (131072, 131072)

# Guardar la matriz
np.save('/media/manuel/Windows/RVITM_estimada.npy', RVITM)

print("RVITM estimada y guardada como: /media/manuel/Windows/RVITM_estimada.npy")
