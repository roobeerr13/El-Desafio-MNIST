
from PIL import Image, ImageOps
import numpy as np
import io

def process_image(image_file):
    """
    Procesa una imagen para que sea compatible con el modelo MNIST.
    - La convierte a escala de grises.
    - Invierte los colores.
    - La redimensiona a 28x28 píxeles.
    - La aplana a un vector de 784 elementos.
    """
    # Leer los bytes de la imagen y abrirla
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Procesar la imagen
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    
    # Convertir a un array de numpy y normalizar
    img_array = np.array(img).astype('float32') / 255.0
    img_array = 1.0 - img_array
    
    # Aplanar la imagen para el modelo
    img_flattened = img_array.reshape(1, 784)
    
    return img_flattened
