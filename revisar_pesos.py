
from pathlib import Path
import torch

ruta = Path(__file__).parent / "mobilenet_v3_small.pth"

checkpoint = torch.load(
    ruta,
    map_location="cpu",
    weights_only=True
)

print("Tipo de archivo:", type(checkpoint))

for clave, valor in checkpoint.items():
    print("\nClave:", clave)
    print("Tipo:", type(valor))

    if isinstance(valor, dict):
        print("Primeras claves:", list(valor.keys())[:10])
    else:
        print("Contenido:", valor)
