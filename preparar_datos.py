
from pathlib import Path
from torchvision.datasets import CIFAR10

# Carpeta local donde se guardará el conjunto de datos
ruta_datos = Path("datos")
ruta_datos.mkdir(exist_ok=True)

print("Preparando el conjunto CIFAR-10...")
print("La primera descarga requiere internet.")

# Descargar el conjunto de entrenamiento
entrenamiento = CIFAR10(
    root=str(ruta_datos),
    train=True,
    download=True
)

# Descargar el conjunto de prueba
prueba = CIFAR10(
    root=str(ruta_datos),
    train=False,
    download=True
)

print("\nDescarga y preparación completadas.")
print("Imágenes de entrenamiento:", len(entrenamiento))
print("Imágenes de prueba:", len(prueba))
print("\nCategorías disponibles:")

for indice, categoria in enumerate(entrenamiento.classes):
    print(indice, "-", categoria)

imagen, etiqueta = entrenamiento[0]

print("\nComprobación de una imagen:")
print("Tamaño:", imagen.size)
print("Categoría:", entrenamiento.classes[etiqueta])
print("\nLos datos están guardados en la carpeta datos.")
