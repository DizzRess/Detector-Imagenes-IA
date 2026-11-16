
# Sistema de escritorio para detectar imágenes generadas por IA con aprendizaje profundo y análisis forense

## Descripción

Este proyecto consiste en el desarrollo de un sistema de escritorio orientado a detectar imágenes generadas por inteligencia artificial (IA), mediante técnicas de aprendizaje profundo y análisis forense de metadatos.

El sistema cuenta con una interfaz gráfica que permite seleccionar imágenes desde el equipo y consultar los resultados de diferentes módulos de análisis.

## Funcionalidades actuales

- **Detección de imágenes:** clasificación de imágenes en las categorías FAKE y REAL mediante una red neuronal MobileNetV3-Small entrenada con el conjunto de datos CIFAKE.
- **Reconocimiento de contenido:** identificación del contenido visual mediante ResNet50, mostrando las cinco predicciones principales.
- **Análisis forense:** extracción de información de la imagen, incluyendo formato, dimensiones, metadatos EXIF disponibles y otros datos técnicos.
- **Identificación digital:** cálculo del hash SHA-256 de la imagen analizada.

## Tecnologías utilizadas

- Python
- PyTorch
- Torchvision
- MobileNetV3-Small
- ResNet50
- Pillow (PIL)
- Tkinter

## Estructura principal

| Archivo | Descripción |
|---|---|
| `interfaz.py` | Interfaz gráfica y ejecución de los módulos del sistema. |
| `mobilenet_cifake_v2.pth` | Pesos del modelo de detección de imágenes FAKE/REAL. |
| `entrenar_mobilenet_cifake_v2.py` | Entrenamiento de MobileNetV3-Small. |
| `evaluar_mobilenet_cifake.py` | Evaluación del modelo. |
| `probar_imagen_cifake.py` | Prueba de clasificación de imágenes. |
| `probar_resnet50.py` | Pruebas del reconocimiento de contenido. |
| `preparar_datos.py` | Preparación de los datos. |

## Requisitos

- Python.
- PyTorch.
- Torchvision.
- Pillow.

## Instalación y ejecución

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/DizzRess/Detector-Imagenes-IA.git
   ```

2. Entrar en la carpeta del proyecto:

   ```bash
   cd Detector-Imagenes-IA
   ```

3. Instalar las dependencias principales:

   ```bash
   pip install torch torchvision pillow
   ```

4. Ejecutar el sistema:

   ```bash
   python interfaz.py
   ```

## Conjunto de datos

El proyecto utiliza CIFAKE para el entrenamiento y evaluación del modelo de detección. El conjunto de datos no se incluye en este repositorio debido a su tamaño.

## Consideraciones

- Los porcentajes mostrados por los modelos representan puntuaciones de predicción y no constituyen garantías de certeza.
- La ausencia o modificación de metadatos no demuestra por sí sola que una imagen haya sido generada por IA.
- El proyecto se encuentra en desarrollo y sus módulos están sujetos a evaluación y mejoras.

## Estado del proyecto

Versión de avance académico. El sistema cuenta con una interfaz gráfica, clasificación FAKE/REAL, reconocimiento de contenido y análisis forense básico de metadatos.

