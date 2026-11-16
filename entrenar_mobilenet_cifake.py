
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from collections import defaultdict

# =====================================
# 1. CONFIGURACIÓN
# =====================================

BASE = Path(__file__).parent
DATASET = BASE / "cifake"
PESOS = BASE / "mobilenet_v3_small.pth"
SALIDA = BASE / "mobilenet_cifake.pth"

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 5

device = torch.device("cpu")
print("Dispositivo:", device)

# =====================================
# 2. TRANSFORMACIONES
# =====================================

transformacion = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# 3. CARGAR CIFAKE
# =====================================

train_data = datasets.ImageFolder(
    DATASET / "train",
    transform=transformacion
)

test_data = datasets.ImageFolder(
    DATASET / "test",
    transform=transformacion
)

print("Categorías:", train_data.classes)
print("Imágenes de entrenamiento:", len(train_data))
print("Imágenes de prueba:", len(test_data))

# Para esta primera prueba utilizaremos
# 10.000 imágenes de cada categoría.
indices_por_clase = defaultdict(list)

for indice, (_, etiqueta) in enumerate(train_data.samples):
    indices_por_clase[etiqueta].append(indice)

indices_train = []

for etiqueta, indices in indices_por_clase.items():
    indices_train.extend(indices[:10000])

train_subset = Subset(train_data, indices_train)

# Utilizamos 2.500 imágenes por clase
# para una evaluación preliminar.
indices_test_por_clase = defaultdict(list)

for indice, (_, etiqueta) in enumerate(test_data.samples):
    indices_test_por_clase[etiqueta].append(indice)

indices_test = []

for etiqueta, indices in indices_test_por_clase.items():
    indices_test.extend(indices[:2500])

test_subset = Subset(test_data, indices_test)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Muestra de entrenamiento:", len(train_subset))
print("Muestra de evaluación:", len(test_subset))

# =====================================
# 4. CARGAR MOBILENETV3-SMALL
# =====================================

modelo = models.mobilenet_v3_small(weights=None)

if not PESOS.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo de pesos: {PESOS}"
    )

checkpoint = torch.load(
    PESOS,
    map_location="cpu",
    weights_only=True
)

print("Claves encontradas:", checkpoint.keys())

if "modelo" in checkpoint:
    pesos_modelo = checkpoint["modelo"]
elif "pesos" in checkpoint:
    pesos_modelo = checkpoint["pesos"]
else:
    raise ValueError("No se encontraron los pesos del modelo.")

resultado = modelo.load_state_dict(
    pesos_modelo,
    strict=True
)

print("Pesos preentrenados cargados:", resultado)

# =====================================
# 5. ADAPTAR A DOS CATEGORÍAS
# =====================================

# Congelar las capas de extracción de características.
for parametro in modelo.features.parameters():
    parametro.requires_grad = False

# Cambiar la última capa: 1000 clases -> 2 clases.
entrada = modelo.classifier[3].in_features

modelo.classifier[3] = nn.Linear(entrada, 2)

modelo = modelo.to(device)

# =====================================
# 6. ENTRENAMIENTO
# =====================================

criterio = nn.CrossEntropyLoss()

optimizador = torch.optim.Adam(
    modelo.classifier.parameters(),
    lr=0.001
)

for epoca in range(EPOCHS):

    modelo.train()
    perdida_total = 0
    aciertos = 0
    total = 0

    for imagenes, etiquetas in train_loader:

        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)

        optimizador.zero_grad()

        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)

        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item()
        predicciones = salidas.argmax(dim=1)

        aciertos += (predicciones == etiquetas).sum().item()
        total += etiquetas.size(0)

    precision_train = 100 * aciertos / total
    perdida_promedio = perdida_total / len(train_loader)

    print(
        f"Época {epoca + 1}/{EPOCHS} | "
        f"Pérdida: {perdida_promedio:.4f} | "
        f"Exactitud entrenamiento: {precision_train:.2f}%"
    )

# =====================================
# 7. EVALUACIÓN PRELIMINAR
# =====================================

modelo.eval()
aciertos = 0
total = 0

with torch.no_grad():
    for imagenes, etiquetas in test_loader:

        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)

        salidas = modelo(imagenes)
        predicciones = salidas.argmax(dim=1)

        aciertos += (predicciones == etiquetas).sum().item()
        total += etiquetas.size(0)

exactitud = 100 * aciertos / total

print("\n========== RESULTADOS ==========")
print(f"Exactitud de evaluación: {exactitud:.2f}%")

# =====================================
# 8. GUARDAR MODELO
# =====================================

torch.save({
    "model_state_dict": modelo.state_dict(),
    "classes": train_data.classes,
    "image_size": IMAGE_SIZE
}, SALIDA)

print("\nModelo guardado en:", SALIDA)
print("Entrenamiento finalizado.")
