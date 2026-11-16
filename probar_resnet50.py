
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


CARPETA = Path(__file__).parent / "imagenes_prueba"

print("Cargando ResNet50...")

pesos = ResNet50_Weights.DEFAULT
modelo = resnet50(weights=pesos)

modelo.eval()

transformacion = pesos.transforms()
categorias = pesos.meta["categories"]

print("Modelo cargado correctamente.")
print("-" * 50)

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

imagenes = [
    archivo for archivo in CARPETA.iterdir()
    if archivo.suffix.lower() in EXTENSIONES
]

for ruta in imagenes:

    print(f"\nImagen: {ruta.name}")

    try:
        imagen = Image.open(ruta).convert("RGB")
        tensor = transformacion(imagen).unsqueeze(0)

        with torch.inference_mode():
            salida = modelo(tensor)
            probabilidades = torch.softmax(salida, dim=1)[0]

        valores, indices = torch.topk(probabilidades, 5)

        print("Cinco predicciones principales:")

        for posicion, (valor, indice) in enumerate(
            zip(valores, indices), start=1
        ):
            nombre = categorias[indice.item()]
            confianza = valor.item() * 100

            print(
                f"{posicion}. {nombre.upper()} "
                f"- {confianza:.2f}%"
            )

    except Exception as error:
        print(f"Error: {error}")

print("\nPrueba finalizada.")
