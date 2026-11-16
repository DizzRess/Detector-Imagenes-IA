
from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# 1. Configuración
DISPOSITIVO = torch.device("cpu")
RUTA_MODELO = Path("modelo_cnn_cifar10.pth")

CATEGORIAS = [
    "avión", "automóvil", "pájaro", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camión"
]


# 2. Definir la misma arquitectura usada en el entrenamiento
class MiCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.caracteristicas = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, imagen):
        x = self.caracteristicas(imagen)
        return self.clasificador(x)


# 3. Preparar la imagen igual que durante el entrenamiento
transformacion = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# 4. Cargar los pesos guardados
if not RUTA_MODELO.exists():
    print("No se encontró el archivo:", RUTA_MODELO)
    sys.exit(1)

modelo = MiCNN().to(DISPOSITIVO)

datos_guardados = torch.load(
    RUTA_MODELO,
    map_location=DISPOSITIVO,
    weights_only=True
)

modelo.load_state_dict(datos_guardados["modelo"])
modelo.eval()


# 5. Obtener la ruta de la imagen
if len(sys.argv) >= 2:
    ruta_imagen = Path(sys.argv[1].strip('"'))
else:
    ruta_imagen = Path(input(
        "Escribe la ruta completa de una imagen: "
    ).strip('"'))

if not ruta_imagen.is_file():
    print("No se encontró la imagen:", ruta_imagen)
    sys.exit(1)


# 6. Predecir la categoría
try:
    with Image.open(ruta_imagen) as imagen_original:
        imagen = imagen_original.convert("RGB")
        tensor = transformacion(imagen).unsqueeze(0)

    with torch.no_grad():
        salidas = modelo(tensor.to(DISPOSITIVO))
        probabilidades = torch.softmax(salidas, dim=1)[0]

    valores, indices = torch.topk(probabilidades, k=3)

    print("\nResultado del reconocimiento")
    print("----------------------------")
    print("Imagen:", ruta_imagen.name)
    print("Categoría principal:", CATEGORIAS[indices[0].item()])
    print(
        "Puntuación del modelo:",
        f"{valores[0].item() * 100:.2f}%"
    )

    print("\nTres categorías principales:")
    for valor, indice in zip(valores, indices):
        print(
            f"- {CATEGORIAS[indice.item()]}: "
            f"{valor.item() * 100:.2f}%"
        )

    print(
        "\nNota: estas puntuaciones son probabilidades normalizadas "
        "del modelo, no una garantía de que la predicción sea correcta."
    )

except Exception as error:
    print("No se pudo analizar la imagen:", error)
