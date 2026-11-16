
from pathlib import Path
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# 1. CONFIGURACIÓN
# =====================================

BASE = Path(__file__).parent
DATASET = BASE / "cifake"
MODELO_PATH = BASE / "mobilenet_cifake.pth"

device = torch.device("cpu")
IMAGE_SIZE = 96

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
# 3. CARGAR DATOS
# =====================================

test_data = datasets.ImageFolder(
    DATASET / "test",
    transform=transformacion
)

indices_por_clase = defaultdict(list)

for indice, (_, etiqueta) in enumerate(test_data.samples):
    indices_por_clase[etiqueta].append(indice)

indices = []

for etiqueta, lista in indices_por_clase.items():
    indices.extend(lista[:2500])

test_subset = Subset(test_data, indices)

test_loader = DataLoader(
    test_subset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# =====================================
# 4. CARGAR MODELO ENTRENADO
# =====================================

checkpoint = torch.load(
    MODELO_PATH,
    map_location=device,
    weights_only=True
)

modelo = models.mobilenet_v3_small(weights=None)

entrada = modelo.classifier[3].in_features
modelo.classifier[3] = torch.nn.Linear(entrada, 2)

modelo.load_state_dict(checkpoint["model_state_dict"])
modelo = modelo.to(device)
modelo.eval()

clases = checkpoint["classes"]

print("Categorías:", clases)

# =====================================
# 5. REALIZAR PREDICCIONES
# =====================================

etiquetas_reales = []
predicciones = []

with torch.no_grad():
    for imagenes, etiquetas in test_loader:

        imagenes = imagenes.to(device)

        salidas = modelo(imagenes)
        resultados = salidas.argmax(dim=1)

        etiquetas_reales.extend(etiquetas.tolist())
        predicciones.extend(resultados.tolist())

# =====================================
# 6. CALCULAR MÉTRICAS
# =====================================

accuracy = accuracy_score(
    etiquetas_reales,
    predicciones
)

precision = precision_score(
    etiquetas_reales,
    predicciones,
    average="binary",
    pos_label=0
)

recall = recall_score(
    etiquetas_reales,
    predicciones,
    average="binary",
    pos_label=0
)

f1 = f1_score(
    etiquetas_reales,
    predicciones,
    average="binary",
    pos_label=0
)

matriz = confusion_matrix(
    etiquetas_reales,
    predicciones,
    labels=[0, 1]
)

print("\n========== MÉTRICAS ==========")
print(f"Exactitud:  {accuracy * 100:.2f}%")
print(f"Precisión FAKE: {precision * 100:.2f}%")
print(f"Recall FAKE:    {recall * 100:.2f}%")
print(f"F1-score FAKE:  {f1 * 100:.2f}%")

print("\n========== MATRIZ DE CONFUSIÓN ==========")
print("Orden de clases:", clases)
print(matriz)

print("\n========== INFORME POR CLASE ==========")
print(classification_report(
    etiquetas_reales,
    predicciones,
    target_names=clases,
    zero_division=0
))
