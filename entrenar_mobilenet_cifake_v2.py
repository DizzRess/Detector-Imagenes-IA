
from pathlib import Path
from collections import defaultdict
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

BASE = Path(__file__).parent
DATASET = BASE / "cifake"
PESOS = BASE / "mobilenet_v3_small.pth"
SALIDA = BASE / "mobilenet_cifake_v2.pth"

SEMILLA = 42
IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 5

random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

device = torch.device("cpu")

print("Dispositivo:", device)
print("Semilla:", SEMILLA)

# ==========================================
# 2. TRANSFORMACIONES
# ==========================================

transformacion = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 3. CARGAR LOS DATOS
# ==========================================

train_data = datasets.ImageFolder(
    DATASET / "train",
    transform=transformacion
)

test_data = datasets.ImageFolder(
    DATASET / "test",
    transform=transformacion
)

clases = train_data.classes

print("Categorías:", clases)

# Seleccionar aleatoriamente 10.000 imágenes
# de cada clase del conjunto de entrenamiento.

def seleccionar_por_clase(dataset, cantidad, semilla):
    grupos = defaultdict(list)

    for indice, (_, etiqueta) in enumerate(dataset.samples):
        grupos[etiqueta].append(indice)

    rng = random.Random(semilla)
    seleccion = []

    for etiqueta in sorted(grupos):
        if len(grupos[etiqueta]) < cantidad:
            raise ValueError("No hay suficientes imágenes por clase.")

        seleccion.extend(
            rng.sample(grupos[etiqueta], cantidad)
        )

    rng.shuffle(seleccion)
    return seleccion


indices_train = seleccionar_por_clase(
    train_data, 10000, SEMILLA
)

# Separar 80% para entrenamiento y 20% para validación.

rng = random.Random(SEMILLA)
rng.shuffle(indices_train)

cantidad_train = int(len(indices_train) * 0.80)

indices_entrenamiento = indices_train[:cantidad_train]
indices_validacion = indices_train[cantidad_train:]

# Seleccionar 2.500 imágenes aleatorias de cada clase
# del conjunto de prueba, que permanece separado.

indices_test = seleccionar_por_clase(
    test_data, 2500, SEMILLA + 1
)

entrenamiento = Subset(
    train_data, indices_entrenamiento
)

validacion = Subset(
    train_data, indices_validacion
)

prueba = Subset(
    test_data, indices_test
)

print("\nDistribución de datos:")
print("Entrenamiento:", len(entrenamiento))
print("Validación:", len(validacion))
print("Prueba:", len(prueba))

# ==========================================
# 4. CARGAR LOS PESOS PREENTRENADOS
# ==========================================

modelo = models.mobilenet_v3_small(weights=None)

checkpoint = torch.load(
    PESOS,
    map_location="cpu",
    weights_only=True
)

modelo.load_state_dict(
    checkpoint["modelo"],
    strict=True
)

print("\nPesos preentrenados cargados correctamente.")

# Congelar las capas de características.

for parametro in modelo.features.parameters():
    parametro.requires_grad = False

# Adaptar la última capa a FAKE y REAL.

entrada = modelo.classifier[3].in_features
modelo.classifier[3] = nn.Linear(entrada, 2)

modelo = modelo.to(device)

# ==========================================
# 5. PREPARAR LOS CARGADORES
# ==========================================

generador = torch.Generator()
generador.manual_seed(SEMILLA)

train_loader = DataLoader(
    entrenamiento,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generador
)

val_loader = DataLoader(
    validacion,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    prueba,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# 6. FUNCIÓN DE EVALUACIÓN
# ==========================================

criterio = nn.CrossEntropyLoss()

def evaluar_perdida(modelo, cargador):
    modelo.eval()
    perdida_total = 0
    total = 0
    aciertos = 0

    with torch.no_grad():
        for imagenes, etiquetas in cargador:
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)

            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)

            perdida_total += perdida.item() * etiquetas.size(0)

            predicciones = salidas.argmax(dim=1)
            aciertos += (
                predicciones == etiquetas
            ).sum().item()

            total += etiquetas.size(0)

    return perdida_total / total, 100 * aciertos / total


# ==========================================
# 7. ENTRENAMIENTO Y VALIDACIÓN
# ==========================================

optimizador = torch.optim.Adam(
    modelo.classifier.parameters(),
    lr=0.001
)

mejor_validacion = -1

print("\n========== ENTRENAMIENTO ==========")

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

        perdida_total += perdida.item() * etiquetas.size(0)

        predicciones = salidas.argmax(dim=1)
        aciertos += (
            predicciones == etiquetas
        ).sum().item()

        total += etiquetas.size(0)

    perdida_train = perdida_total / total
    accuracy_train = 100 * aciertos / total

    perdida_val, accuracy_val = evaluar_perdida(
        modelo, val_loader
    )

    print(
        f"Época {epoca + 1}/{EPOCHS} | "
        f"Pérdida entrenamiento: {perdida_train:.4f} | "
        f"Exactitud entrenamiento: {accuracy_train:.2f}% | "
        f"Pérdida validación: {perdida_val:.4f} | "
        f"Exactitud validación: {accuracy_val:.2f}%"
    )

    # Guardar el modelo con mejor exactitud de validación.

    if accuracy_val > mejor_validacion:

        mejor_validacion = accuracy_val

        torch.save({
            "model_state_dict": modelo.state_dict(),
            "classes": clases,
            "image_size": IMAGE_SIZE,
            "seed": SEMILLA
        }, SALIDA)

        print("  -> Nuevo mejor modelo guardado.")

# ==========================================
# 8. EVALUACIÓN FINAL CON TEST
# ==========================================

print("\n========== EVALUACIÓN FINAL ==========")

mejor_checkpoint = torch.load(
    SALIDA,
    map_location=device,
    weights_only=True
)

modelo.load_state_dict(
    mejor_checkpoint["model_state_dict"]
)

modelo.eval()

etiquetas_reales = []
predicciones = []

with torch.no_grad():
    for imagenes, etiquetas in test_loader:

        imagenes = imagenes.to(device)

        salidas = modelo(imagenes)
        resultados = salidas.argmax(dim=1)

        etiquetas_reales.extend(etiquetas.tolist())
        predicciones.extend(resultados.tolist())

accuracy = accuracy_score(
    etiquetas_reales, predicciones
)

precision = precision_score(
    etiquetas_reales, predicciones,
    pos_label=0
)

recall = recall_score(
    etiquetas_reales, predicciones,
    pos_label=0
)

f1 = f1_score(
    etiquetas_reales, predicciones,
    pos_label=0
)

matriz = confusion_matrix(
    etiquetas_reales, predicciones,
    labels=[0, 1]
)

print(f"Mejor exactitud de validación: {mejor_validacion:.2f}%")
print(f"Exactitud de prueba: {accuracy * 100:.2f}%")
print(f"Precisión FAKE: {precision * 100:.2f}%")
print(f"Recall FAKE: {recall * 100:.2f}%")
print(f"F1-score FAKE: {f1 * 100:.2f}%")

print("\nMatriz de confusión:")
print("Orden:", clases)
print(matriz)

print("\nInforme por clase:")
print(classification_report(
    etiquetas_reales,
    predicciones,
    target_names=clases,
    zero_division=0
))

print("\nModelo guardado en:", SALIDA)
print("Experimento finalizado.")
