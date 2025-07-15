from PIL import Image
import os

def resize_and_binarize(input_path, output_path, size=(64, 64), threshold=128):
    # Abrir imagen original
    image = Image.open(input_path)

    # Convertir a escala de grises
    gray_image = image.convert("L")

    # Redimensionar
    resized_image = gray_image.resize(size, Image.BICUBIC)

    # Binarizar: píxeles > threshold se vuelven 255 (blanco), el resto 0 (negro)
    binary_image = resized_image.point(lambda p: 255 if p > threshold else 0, mode='1')

    # Guardar imagen binarizada
    binary_image.save(output_path)
    print(f"Imagen binaria guardada en: {output_path}")

# === Parámetros ===
input_image = "/home/manuel/Documents/Fibras_opticas/Proyecto_Final/Input_Images/microscopio.png"
output_image = "/home/manuel/Documents/Fibras_opticas/Proyecto_Final/Resized_Images/microscopio_64x64_bw.png"

resize_and_binarize(input_image, output_image)
