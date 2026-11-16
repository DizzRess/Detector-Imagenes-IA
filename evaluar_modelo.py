
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DISPOSITIVO = torch.device("cpu")
RUTA_MODELO = "modelo_cnn_cifar10.pth"
RUTA_DATOS = "datos"

CATEGORIAS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Debe coincidir con la arquitectura utilizada durante el entrenamiento.
class ModeloCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.caracteristicas = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
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

    def forward(self, x):
        x = self.caracteristicas(x)
        return self.clasificador(x)


transformaciones = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

datos = datasets.CIFAR10(
    root=RUTA_DATOS,
    train=False,
    download=False,
    transform=transformaciones
)

cargador = DataLoader(
    datos,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

modelo = ModeloCNN().to(DISPOSITIVO)

checkpoint = torch.load(
    RUTA_MODELO,
    map_location=DISPOSITIVO,
    weights_only=True
)

modelo.load_state_dict(checkpoint["modelo"])
modelo.eval()

correctas_por_clase = [0] * 10
total_por_clase = [0] * 10
total_correctas = 0
total_imagenes = 0

with torch.no_grad():
    for imagenes, etiquetas in cargador:
        imagenes = imagenes.to(DISPOSITIVO)
        etiquetas = etiquetas.to(DISPOSITIVO)

        salidas = modelo(imagenes)
        predicciones = salidas.argmax(dim=1)

        total_correctas += (predicciones == etiquetas).sum().item()
        total_imagenes += etiquetas.size(0)

        for i in range(10):
            mascara = etiquetas == i
            total_por_clase[i] += mascara.sum().item()
            correctas_por_clase[i] += (
                (predicciones == etiquetas) & mascara
            ).sum().item()

precision = 100 * total_correctas / total_imagenes

print("\n===== EVALUACIÓN DEL MODELO =====")
print(f"Imágenes evaluadas: {total_imagenes}")
print(f"Predicciones correctas: {total_correctas}")
print(f"Precisión general: {precision:.2f}%")

print("\n===== RESULTADOS POR CATEGORÍA =====")

for i, categoria in enumerate(CATEGORIAS):
    aciertos = correctas_por_clase[i]
    total = total_por_clase[i]
    porcentaje = 100 * aciertos / total if total else 0

    print(
        f"{categoria}: {aciertos}/{total} "
        f"({porcentaje:.2f}%)"
    )

print("\nEvaluación finalizada.")
