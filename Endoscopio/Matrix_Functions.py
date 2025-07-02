import numpy as np
import scipy 
import pygame
import scipy.linalg
import os
import matplotlib.pyplot as plt

def pattern_adapt(matrix_NotAdapated, convertion_Column, DMD_pixels = [1920,1080]):
    '''
    Esta función recibe una matriz de orden N^2xN^2 como primer argumento
    y retrona una matriz NxN que contiene solamente
    los píxeles de la columna respectiva al segundo argumento
    '''
    if (len(matrix_NotAdapated) & (len(matrix_NotAdapated)- 1) != 0):
        #Si es diferente de 0, entonces no es potencia de dos, no es útil para nuestro trabajo
        return print("La matriz ingresada debe ser potencia de dos")    #Early return para atrapar un error común
    if (len(matrix_NotAdapated) != len(matrix_NotAdapated[0])):
        return print("La matriz no es cuadrada")                #Otra verificación rápida
              
    matrix_Adapted = []                                         #Se define la matriz que se va a devolver
    aux_matrix = []                                             #Utilizaremos esta matriz auxiliar para trabajar en los ciclos
    square_aux_matrix = []                                      #Matriz intermedia que contendrá el patrón de una columna en un cuadrado
    row_count = int(np.sqrt(len(matrix_NotAdapated)))           #Requerimos saber el tamaño N del patrón NxN que se va a adpatar
    scale_Factor = int(np.min(DMD_pixels)/row_count)            #Para escalar los píxeles
    offset = 0             
    '''Primero lo que haremos es convertir una columna de la matriz origen en una matriz cuadrada
        Llevando la lógica de:
            Patrón Proyectado   :     Patrón origen
                fila [0]        :   Columna [0-63]
                fila [1]        :   Columna [64-127]
                ...
                fila [64]       :   Columna [4032-4095]'''
    
    for row_Adapted in range(0, row_count,1):
        #Se hace un ciclo de N paso para asignar N filas
        for row_NotAdapted in range(offset, row_count + offset,1):
            #Se añade cada elemento de la columna a una fila
            aux_matrix.append(matrix_NotAdapated[row_NotAdapted, convertion_Column])
        offset+= row_count                         #Empezamos el siguiente ciclo desde nos quedamos en el anterior
        square_aux_matrix.append(aux_matrix)       #Añadimos una fila a la matriz cuadrada
        aux_matrix = []                            #Se limpian los elementos de la matriz
    #En este punto square_aux_matrix ya es una matriz NxN 

    ''' Se deben adapatar las matrices de NxN píxeles a 1920x1080 píxeles
        Se buscará conservar el patrón en un cuadrado '''
    for row in range(0, row_count, 1): 
        for rows_scale in range(0, scale_Factor, 1):
            for column in range(0, row_count, 1):
                for column_scale in range(0,scale_Factor,1):
                    aux_matrix.append(square_aux_matrix[row][column])   #Lista de listas
            matrix_Adapted.append(aux_matrix)
            aux_matrix = []
    return matrix_Adapted


def embed_in_DMD_frame(matrix, target_size=(1920, 1080)):
    """
    Centra la matriz  dentro de un lienzo negro del tamaño `target_size` = (ancho, alto)
    """
    pattern = np.array(matrix, dtype=np.uint8)
    pattern_height, pattern_width = pattern.shape
    target_width, target_height = target_size
    canvas = np.zeros((target_width, target_height), dtype=np.uint8)

    # Centrado 
    x_offset = (target_width - pattern_width) // 2
    y_offset = (target_height - pattern_height) // 2

    canvas[ x_offset:x_offset + pattern_width,y_offset:y_offset + pattern_height] = pattern
    return canvas


def get_Hadamard_1_SurfaceToShow(pattern_size, convertion_Column):
    '''     
    Genera una superficie (surface) de Pygame correspondiente a un patrón de Hadamard específico,
    adaptado para ser proyectado en un DMD de resolución 1920x1080.

    Parámetros:
    - pattern_size: int
        Tamaño del patrón cuadrado deseado (ej. 64, 128). La matriz Hadamard generada será de orden pattern_size^2.
    - convertion_Column: int
        Índice de la columna de la matriz Hadamard original que se desea proyectar como patrón.

    Proceso:
    1. Genera una matriz de Hadamard de orden (pattern_size^2 × pattern_size^2).
    2. Convierte todos los valores de la matriz Hadamard de (-1, 1) a (0, 255), para ser interpretados como escala de grises.
    3. Extrae la columna deseada de la matriz Hadamard y la reorganiza como una matriz cuadrada (pattern_size × pattern_size).
    4. Escala esa matriz cuadrada para llenar un área proporcional dentro del DMD.
    5. Centra el patrón en un lienzo negro del tamaño del DMD (1080 × 1920).
    6. Convierte la imagen 2D en una imagen RGB, necesaria para su visualización con Pygame.
    7. Crea y retorna una superficie de Pygame lista para ser proyectada.

    Retorna:
    - Hadamard_H1_surface: pygame.Surface
        Superficie con el patrón Hadamard listo para proyectar en pantalla.
    '''
    matrix_order = pattern_size**2
    Hadamard_Matrix = scipy.linalg.hadamard(matrix_order)                           # Se crea una matriz de Hadamard del orden previamente establecido
    Hadamard_H1 = ((Hadamard_Matrix + 1) / 2 * 255).astype(np.uint8)                # Valores 0 a 255
    Hadamard_H1_Scaled = pattern_adapt(Hadamard_H1, convertion_Column)              # Se selecciona el patrón n-ésimo
    Hadamard_H1_Centered = embed_in_DMD_frame(Hadamard_H1_Scaled)                   # Se escala el patrón para mostrarlo en el DMD
    Hadamard_H1_rgb_pattern = np.stack([Hadamard_H1_Centered]*3, axis=-1)           # Se adecua el patrón para convertirlo en formate RGB ( R,G,B = 255 (Blanco)) para su proyección
    Hadamard_H1_surface = pygame.surfarray.make_surface(Hadamard_H1_rgb_pattern)    # Se convierte en superficie el patrón para mostrarlo con pygame
    return Hadamard_H1_surface

def get_Hadamard_2_SurfaceToShow(pattern_size, convertion_Column):
    # Generar matriz Hadamard, esta es la que se debe usar para la recuperación de la imagen
    matrix_order = pattern_size**2
    Hadamard_Matrix = scipy.linalg.hadamard(matrix_order)                           # Se crea una matriz de Hadamard del orden previamente establecido
    Hadamard_H2 = ((Hadamard_Matrix - 1) / 2 * 255).astype(np.uint8)                # Valores 0 a 255
    Hadamard_H2_Scaled = pattern_adapt(Hadamard_H2, convertion_Column)              # Se selecciona el patrón n-ésimo
    Hadamard_H2_Centered = embed_in_DMD_frame(Hadamard_H2_Scaled)                   # Se escala el patrón para mostrarlo en el DMD
    Hadamard_H2_rgb_pattern = np.stack([Hadamard_H2_Centered]*3, axis=-1)           # Se adecua el patrón para convertirlo en formate RGB ( R,G,B = 255 (Blanco)) para su proyección
    Hadamard_H2_surface = pygame.surfarray.make_surface(Hadamard_H2_rgb_pattern)    # Se convierte en superficie el patrón para mostrarlo con pygame
    return Hadamard_H2_surface

def save_hadamard_patterns_to_disk(pattern_size, Hadmard_Selection ,output_dir="Hadamard_Patterns"):
    """
    Generates and saves to disk all Hadamard patterns (as PNG images)
    for a given pattern resolution.

    Parameters:
    - pattern_size: int
        Size N of the square pattern (e.g., 64, 128). N^2 patterns will be generated.
    - output_dir: str
        Directory where the patterns will be saved as PNG images.
    """
    os.makedirs(output_dir, exist_ok=True)
    matrix_order = pattern_size ** 2
    print(f"Generating Hadamard matrix of order {matrix_order}...")

    # Generate Hadamard matrix
    hadamard_matrix = scipy.linalg.hadamard(matrix_order)
    if (Hadmard_Selection == 1):
        hadamard_matrix = ((hadamard_matrix + 1) / 2 * 255).astype(np.uint8)
    else:
        hadamard_matrix = ((hadamard_matrix - 1) / 2 * 255).astype(np.uint8)

    for i in range(matrix_order):
        pattern_scaled = pattern_adapt(hadamard_matrix, i)
        pattern_centered = embed_in_DMD_frame(pattern_scaled)

        # Save as PNG image
        filename = os.path.join(output_dir, f"hadamard_{i:04}.png")
        plt.imsave(filename, pattern_centered.T, cmap="gray", vmin=0, vmax=255)

        if i % 100 == 0 or i == matrix_order - 1:
            print(f"Saved pattern {i + 1} of {matrix_order}")

    print(f"All patterns saved in folder: '{output_dir}/'")

def display_hadamard_patterns_from_disk(folder_path, fps=60):
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    
    running = True
    index = 0
    while running and index < len(filenames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                running = False

        # Carga imagen individual
        path = os.path.join(folder_path, filenames[index])
        image = pygame.image.load(path).convert()

        screen.blit(image, (0, 0))
        pygame.display.flip()
        clock.tick(fps)
        index += 1

    pygame.quit()


#save_hadamard_patterns_to_disk(64,1, "Hadmard_2")