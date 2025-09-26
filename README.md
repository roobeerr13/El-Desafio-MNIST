# El-Desafio-MNIST

Este proyecto es una aplicación web para el reconocimiento de dígitos manuscritos. Utiliza un modelo de aprendizaje profundo entrenado en el famoso conjunto de datos MNIST. La aplicación permite a los usuarios dibujar un dígito en una interfaz web, y el modelo predecirá qué dígito es.

El backend está construido con Flask en Python y utiliza una red neuronal convolucional (CNN) creada con Keras/TensorFlow para realizar las predicciones.

## Características

*   **Interfaz de dibujo interactiva:** Dibuja dígitos directamente en el navegador.
*   **Procesamiento de imágenes en tiempo real:** La imagen dibujada se procesa y se envía al modelo para su predicción.
*   **Modelo de alta precisión:** Utiliza una CNN entrenada en MNIST para lograr una alta precisión en el reconocimiento de dígitos.
*   **Fácil de usar:** Interfaz simple e intuitiva.

## Tecnologías Utilizadas

*   **Backend:**
    *   Flask
    *   TensorFlow / Keras
    *   NumPy
    *   Pillow
*   **Frontend:**
    *   HTML
    *   CSS
    *   JavaScript

## Estructura del Proyecto

```
.
├── main.py                # Aplicación principal de Flask
├── mnist_trainer.py         # Script para entrenar el modelo CNN
├── mnist_model.h5           # Modelo pre-entrenado
├── requirements.txt         # Dependencias de Python
├── src
│   ├── index.html           # Interfaz de usuario web
│   └── decoracion.css       # Estilos para la interfaz
└── static
    ├── accuracy.png         # Gráfico de precisión del entrenamiento
    └── loss.png             # Gráfico de pérdida del entrenamiento
```

## Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/El-Desafio-MNIST.git
    cd El-Desafio-MNIST
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1.  **Ejecuta la aplicación Flask:**
    ```bash
    python main.py
    ```

2.  **Abre tu navegador:**
    Ve a `http://127.0.0.1:5000` en tu navegador web.

3.  **Dibuja un dígito:**
    Usa el lienzo para dibujar un dígito del 0 al 9.

4.  **Obtén la predicción:**
    La aplicación mostrará el dígito que el modelo ha predicho.

## Contribuciones

Las contribuciones son bienvenidas. Si tienes alguna idea o sugerencia, no dudes in abrir un *issue* o enviar un *pull request*.
