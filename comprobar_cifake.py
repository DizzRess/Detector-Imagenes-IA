from pathlib import Path
from torchvision import datasets, transforms

# Ubicación del conjunto de datos
ruta = Path(__file__).parent / "cifake"

# Transformación de las imágenes
transformacion = transforms.ToTensor()

# Cargar los conjuntos
entrenamiento = datasets.ImageFolder(
    ruta / "train",
    transform=transformacion
)

prueba = datasets.ImageFolder(
    ruta / "test",
    transform=transformacion
)

# Mostrar información
print("========== CIFAKE ==========")

print("\nCategorías:", entrenamiento.classes)
print("Distribución de etiquetas:", entrenamiento.class_to_idx)

print("\nImágenes de entrenamiento:")
print("Total:", len(entrenamiento))

for clase, cantidad in zip(
    entrenamiento.classes,
    [sum(1 for _, etiqueta in entrenamiento.samples
         if etiqueta == indice)
     for indice in range(len(entrenamiento.classes))]
):
    print(f"{clase}: {cantidad}")

print("\nImágenes de prueba:")
print("Total:", len(prueba))

for clase, cantidad in zip(
    prueba.classes,
    [sum(1 for _, etiqueta in prueba.samples
         if etiqueta == indice)
     for indice in range(len(prueba.classes))]
):
    print(f"{clase}: {cantidad}")

# Comprobar que una imagen se puede cargar
imagen, etiqueta = entrenamiento[0]

print("\n========== PRUEBA DE CARGA ==========")
print("Imagen cargada correctamente.")
print("Dimensiones:", imagen.shape)
print("Categoría:", entrenamiento.classes[etiqueta])