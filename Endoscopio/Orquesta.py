import pygame
from instrumental.drivers.cameras import uc480
import imageio
import time
import os

# Configuración Cámara
num_frames = 4096
delay = 0.05  # segundos entre capturas
exposure_time_ms = 5  # tiempo de exposición (ms)
gain_value = 0        # ganancia (0 a 100)
output_dir = "Speckles_Reconstruir" #Speckles

# Crear carpeta si no existe
os.makedirs(output_dir, exist_ok = True)
# Inicializar cámara
cams = uc480.list_instruments()
cam = uc480.UC480_Camera(cams[0])

# Establecer parámetros
cam.exposure_time = exposure_time_ms 
cam.gain = gain_value

pygame.init()   #Iniciamos la instancia de pygame que nosz permite proyectar los patrones
screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN) #Configuración para la proyección 
clock = pygame.time.Clock() #Creamos una instancia para poder seleccionar los fps de proyección
    
# Iniciar captura 
cam.start_live_video(framerate = "30Hz")


fps = 30 #Frames por segundo para la proyección
print(f"Iniciando captura de {num_frames} imágenes en '{output_dir}/'...")


running = True  #Variable para poder apagar en algún momento la proyección
index = 0       #Variable para poder iterar sobre las imágenes
contador_frame = 0

folder_path = folder_path = "D:\\Hadamard_1_64_1280x1024"
filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])  #Ji
while running and index < len(filenames):   
    for event in pygame.event.get():
        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
            running = False

    # Carga imagen individual
    path = os.path.join(folder_path, filenames[index])
    image = pygame.image.load(path)

    screen.blit(image,  (320,0))
    pygame.display.flip()   #En esta línea se proyecta la imagen en pantalla
    
    time.sleep(delay)
    frame = cam.grab_image(timeout="2s", copy=True)
    filename = os.path.join(output_dir, f"frame_{contador_frame:02d}.png")
    imageio.imwrite(filename, frame)
    #print(f"Imagen {contador_frame+1}/{num_frames} guardada como {filename}")
    contador_frame+=1
    
    index += 1

folder_path = "D:\\Hadamard_1_64_1280x1024"
index = 0
while running and index < len(filenames):   
    for event in pygame.event.get():
        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
            running = False

    # Carga imagen individual
    path = os.path.join(folder_path, filenames[index])
    image = pygame.image.load(path)

    screen.blit(image, (320,0))
    pygame.display.flip()   #En esta línea se proyecta la imagen en pantalla
    
    time.sleep(delay)
    frame = cam.grab_image(timeout="2s", copy=True)
    filename = os.path.join(output_dir, f"frame_{contador_frame:02d}.png")
    imageio.imwrite(filename, frame)
    #print(f"Imagen {contador_frame+1}/{num_frames} guardada como {filename}")
    contador_frame+=1
    
    index += 1


#   Proyectar Imágenes para reconstruir  
folder_path = "D:\\Imagenes_Reconstruir_1280x1024"
filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
index = 0
while running and index < len(filenames):   
    for event in pygame.event.get():
        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
            running = False

    # Carga imagen individual
    path = os.path.join(folder_path, filenames[index])
    image = pygame.image.load(path)

    screen.blit(image,  (320,0))
    pygame.display.flip()   #En esta línea se proyecta la imagen en pantalla
    
    time.sleep(0.5)
    frame = cam.grab_image(timeout="2s", copy=True)
    filename = os.path.join(output_dir, f"frame_{contador_frame:02d}.png")
    imageio.imwrite(filename, frame)
    #print(f"Imagen {contador_frame+1}/{num_frames} guardada como {filename}")
    contador_frame+=1
    
    index += 1    
    
cam.stop_live_video()
pygame.quit()
print("Captura completada.")