

\# Sistema de escritorio para detectar imágenes generadas por IA con aprendizaje profundo y análisis forense



\## Descripción



Este proyecto consiste en el desarrollo de un sistema de escritorio orientado a detectar imágenes generadas por inteligencia artificial (IA), mediante técnicas de aprendizaje profundo y análisis forense de metadatos.



El sistema cuenta con una interfaz gráfica que permite seleccionar imágenes desde el equipo y consultar los resultados de diferentes módulos de análisis.



\## Funcionalidades actuales



\- \*\*Detección de imágenes:\*\* clasificación de imágenes en las categorías FAKE y REAL mediante una red neuronal MobileNetV3-Small entrenada con el conjunto de datos CIFAKE.

\- \*\*Reconocimiento de contenido:\*\* identificación del contenido visual mediante ResNet50, mostrando las cinco predicciones principales.

\- \*\*Análisis forense:\*\* extracción de información de la imagen, incluyendo formato, dimensiones, metadatos EXIF disponibles y otros datos técnicos.

\- \*\*Identificación digital:\*\* cálculo del hash SHA-256 de la imagen analizada.



\## Tecnologías utilizadas



\- Python

\- PyTorch

\- Torchvision

\- MobileNetV3-Small

\- ResNet50

\- Pillow (PIL)

\- Tkinter



\## Estructura principal



| Archivo | Descripción |

|---|---|

| `interfaz.py` | Interfaz gráfica y ejecución de los módulos del sistema. |

| `mobilenet\_cifake\_v2.pth` | Pesos del modelo de detección de imágenes FAKE/REAL. |

| `entrenar\_mobilenet\_cifake\_v2.py` | Entrenamiento de MobileNetV3-Small. |

| `evaluar\_mobilenet\_cifake.py` | Evaluación del modelo. |

| `probar\_imagen\_cifake.py` | Prueba de clasificación de imágenes. |

| `probar\_resnet50.py` | Pruebas del reconocimiento de contenido. |

| `preparar\_datos.py` | Preparación de los datos. |



\## Requisitos



\- Python instalado.

\- PyTorch.

\- Torchvision.

\- Pillow.



\## Instalación y ejecución



1\. Clonar el repositorio:



&#x20;  ```bash

&#x20;  git clone https://github.com/DizzRess/Detector-Imagenes-IA.git

&#x20;  ```



2\. Entrar en la carpeta del proyecto:



&#x20;  ```bash

&#x20;  cd Detector-Imagenes-IA

&#x20;  ```



3\. Instalar las dependencias principales:



&#x20;  ```bash

&#x20;  pip install torch torchvision pillow

&#x20;  ```



4\. Ejecutar el sistema:



&#x20;  ```bash

&#x20;  python interfaz.py

&#x20;  ```



\## Conjunto de datos



El proyecto utiliza CIFAKE para el entrenamiento y evaluación del modelo de detección. El conjunto de datos no se incluye en este repositorio debido a su tamaño.



\## Consideraciones



\- Los porcentajes mostrados por los modelos representan puntuaciones de predicción y no constituyen garantías de certeza.

\- La ausencia o modificación de metadatos no demuestra por sí sola que una imagen haya sido generada por IA.

\- El proyecto se encuentra en desarrollo y sus módulos están sujetos a evaluación y mejoras.



\## Estado del proyecto



Versión de avance académico. El sistema cuenta con una interfaz gráfica, clasificación FAKE/REAL, reconocimiento de contenido y análisis forense básico de metadatos.



