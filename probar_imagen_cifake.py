
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configuración
CARPETA = Path(__file__).parent
PESOS = CARPETA / "mobilenet_cifake_v2.pth"

DISPOSITIVO = torch.device("cpu")
CATEGORIAS = ["FAKE", "REAL"]

# Transformaciones: deben coincidir con el entrenamiento
transformacion = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Crear arquitectura
modelo = models.mobilenet_v3_small(weights=None)
modelo.classifier[3] = nn.Linear(
    modelo.classifier[3].in_features, 2
)

# Cargar modelo entrenado
checkpoint = torch.load(
    PESOS,
    map_location=DISPOSITIVO,
    weights_only=True
)

modelo.load_state_dict(checkpoint["model_state_dict"], strict=True)
modelo.to(DISPOSITIVO)
modelo.eval()

def predecir(ruta_imagen):
    imagen = Image.open(ruta_imagen).convert("RGB")
    tensor = transformacion(imagen).unsqueeze(0)

    with torch.no_grad():
        salida = modelo(tensor)
        probabilidades = torch.softmax(salida, dim=1)[0]

    indice = probabilidades.argmax().item()
    categoria = CATEGORIAS[indice]
    confianza = probabilidades[indice].item() * 100

    print("\n========== RESULTADO ==========")
    print(f"Imagen: {ruta_imagen.name}")
    print(f"Clasificación: {categoria}")
    print(f"Confianza: {confianza:.2f}%")
    print("\nProbabilidades:")

    for i, nombre in enumerate(CATEGORIAS):
        print(f"{nombre}: {probabilidades[i].item() * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python probar_imagen_cifake.py ruta_de_imagen")
    else:
        ruta = Path(sys.argv[1])
        if not ruta.is_file():
            print("No se encontró la imagen:", ruta)
        else:
            predecir(ruta)
