from pathlib import Path
import torch

ruta = Path(__file__).parent / "mobilenet_cifake_v2.pth"

checkpoint = torch.load(
    ruta,
    map_location="cpu",
    weights_only=True
)

print("Tipo de archivo:", type(checkpoint))

if isinstance(checkpoint, dict):
    for clave, valor in checkpoint.items():
        print("\nClave:", clave)
        print("Tipo:", type(valor))

        if isinstance(valor, dict):
            print("Primeras claves:", list(valor.keys())[:10])
        else:
            print("Contenido:", valor)