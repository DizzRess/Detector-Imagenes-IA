
import torch
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

RUTA_MODELO = "mobilenet_v3_small.pth"

print("Descargando MobileNetV3 Small...")
print("La primera descarga necesita Internet.")

# Descarga los pesos preentrenados oficiales.
pesos = MobileNet_V3_Small_Weights.DEFAULT
modelo = mobilenet_v3_small(weights=pesos)

# Guardamos el modelo y sus categorías en el proyecto.
torch.save(
    {
        "modelo": modelo.state_dict(),
        "categorias": pesos.meta["categories"],
        "nombre": "MobileNetV3 Small",
        "pesos": "ImageNet-1K"
    },
    RUTA_MODELO
)

print("\nModelo guardado correctamente.")
print(f"Archivo: {RUTA_MODELO}")
print(f"Categorías disponibles: {len(pesos.meta['categories'])}")
print("Ya puedes preparar el siguiente paso: usarlo sin Internet.")
