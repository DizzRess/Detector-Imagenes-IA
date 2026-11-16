
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

RUTA_MODELO = Path("mobilenet_v3_small.pth")
DISPOSITIVO = torch.device("cpu")

# Verificar que el modelo local exista.
if not RUTA_MODELO.exists():
    print("Error: no se encuentra mobilenet_v3_small.pth")
    sys.exit(1)

# Cargar el modelo sin descargar pesos.
checkpoint = torch.load(
    RUTA_MODELO,
    map_location=DISPOSITIVO,
    weights_only=True
)

modelo = mobilenet_v3_small(weights=None)
modelo.load_state_dict(checkpoint["modelo"])
modelo.eval()

categorias = checkpoint["categorias"]

# Preparación de imágenes recomendada para los pesos ImageNet.
transformaciones = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

def reconocer(ruta_imagen):
    ruta = Path(ruta_imagen)

    if not ruta.is_file():
        print(f"Error: no se encuentra la imagen: {ruta}")
        return

    imagen = Image.open(ruta).convert("RGB")
    entrada = transformaciones(imagen).unsqueeze(0)

    with torch.no_grad():
        salida = modelo(entrada)
        probabilidades = torch.nn.functional.softmax(salida, dim=1)[0]
        valores, indices = torch.topk(probabilidades, 3)

    print("\n===== RESULTADO DEL RECONOCIMIENTO =====")
    print(f"Imagen: {ruta.name}")
    print("Tres categorías principales:")

    for valor, indice in zip(valores, indices):
        categoria = categorias[indice.item()]
        puntuacion = valor.item() * 100
        print(f"- {categoria}: {puntuacion:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predecir_mobilenet.py nombre_imagen.jpg")
    else:
        reconocer(sys.argv[1])
