from instrumental.drivers.cameras import uc480
import imageio
import time
import os

# Configuración
num_frames = 2
delay = 0.5  # segundos entre capturas
exposure_time_ms = 50  # tiempo de exposición (ms)
gain_value = 100        # ganancia (0 a 100)
output_dir = "Speckles"

# Crear carpeta si no existe
os.makedirs(output_dir, exist_ok = True)

# Inicializar cámara
cams = uc480.list_instruments()
cam = uc480.UC480_Camera(cams[0])

# Establecer parámetros
cam.exposure_time = exposure_time_ms 
cam.gain = gain_value

# Iniciar captura
cam.start_live_video(framerate = "30Hz")

print(f"Iniciando captura de {num_frames} imágenes en '{output_dir}/'...")

for i in range(num_frames):
    frame = cam.grab_image(timeout="2s", copy=True)
    filename = os.path.join(output_dir, f"frame_{i:02d}.png")
    imageio.imwrite(filename, frame)
    print(f"Imagen {i+1}/{num_frames} guardada como {filename}")
    time.sleep(delay)

cam.stop_live_video()
print("Captura completada.")
