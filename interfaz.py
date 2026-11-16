
from pathlib import Path
import hashlib
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
from PIL.ExifTags import TAGS

import torch
import torch.nn as nn
from torchvision import models, transforms


# ==========================================
# CONFIGURACIÓN
# ==========================================

CARPETA = Path(__file__).parent
PESOS_CIFAKE = CARPETA / "mobilenet_cifake_v2.pth"

DISPOSITIVO = torch.device("cpu")

modelo_cifake = None
modelo_resnet = None
transformacion_resnet = None
categorias_resnet = []

CATEGORIAS_CIFAKE = ["FAKE", "REAL"]
TAMANO_IMAGEN = 96

ruta_seleccionada = None
imagen_mostrada = None


# ==========================================
# CARGAR MODELOS
# ==========================================

def cargar_modelos():
    global modelo_cifake, modelo_resnet
    global CATEGORIAS_CIFAKE, TAMANO_IMAGEN
    global transformacion_resnet, categorias_resnet

    # Modelo 1: detección REAL / FAKE con CIFAKE
    checkpoint = torch.load(
        PESOS_CIFAKE,
        map_location=DISPOSITIVO,
        weights_only=True
    )

    CATEGORIAS_CIFAKE = checkpoint.get(
        "classes", ["FAKE", "REAL"]
    )

    TAMANO_IMAGEN = checkpoint.get("image_size", 96)

    modelo_cifake = models.mobilenet_v3_small(weights=None)

    modelo_cifake.classifier[3] = nn.Linear(
        modelo_cifake.classifier[3].in_features,
        len(CATEGORIAS_CIFAKE)
    )

    modelo_cifake.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    modelo_cifake.to(DISPOSITIVO)
    modelo_cifake.eval()

    # Modelo 2: reconocimiento del contenido con ResNet50
    pesos_resnet = models.ResNet50_Weights.DEFAULT
    modelo_resnet = models.resnet50(weights=pesos_resnet)
    modelo_resnet.to(DISPOSITIVO)
    modelo_resnet.eval()

    transformacion_resnet = pesos_resnet.transforms()
    categorias_resnet = pesos_resnet.meta["categories"]


# ==========================================
# TRANSFORMACIÓN CIFAKE
# ==========================================

def transformar_imagen_cifake(imagen):
    transformacion = transforms.Compose([
        transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transformacion(imagen).unsqueeze(0)


# ==========================================
# SELECCIONAR IMAGEN
# ==========================================

def seleccionar_imagen():
    global ruta_seleccionada, imagen_mostrada

    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Todos los archivos", "*.*")
        ]
    )

    if not ruta:
        return

    try:
        imagen = Image.open(ruta).convert("RGB")

        vista = imagen.copy()
        vista.thumbnail((500, 330))

        imagen_mostrada = ImageTk.PhotoImage(vista)

        etiqueta_imagen.config(
            image=imagen_mostrada,
            text=""
        )

        ruta_seleccionada = ruta

        etiqueta_archivo.config(text=Path(ruta).name)

        etiqueta_resultado.config(
            text="Imagen pendiente de análisis",
            fg="#34495e"
        )

        etiqueta_confianza.config(text="")

        texto_contenido.config(state="normal")
        texto_contenido.delete("1.0", tk.END)
        texto_contenido.insert(
            tk.END,
            "Todavía no se ha reconocido el contenido de la imagen."
        )
        texto_contenido.config(state="disabled")

        texto_forense.config(state="normal")
        texto_forense.delete("1.0", tk.END)
        texto_forense.insert(
            tk.END,
            "El informe forense aparecerá aquí después del análisis."
        )
        texto_forense.config(state="disabled")

        boton_analizar.config(state="normal")

    except Exception as error:
        messagebox.showerror(
            "Error",
            f"No se pudo abrir la imagen:\n{error}"
        )


# ==========================================
# DETECCIÓN REAL / FAKE
# ==========================================

def detectar_origen(imagen):
    tensor = transformar_imagen_cifake(imagen)

    with torch.inference_mode():
        salida = modelo_cifake(tensor)
        probabilidades = torch.softmax(salida, dim=1)[0]

    indice = probabilidades.argmax().item()
    categoria = CATEGORIAS_CIFAKE[indice]
    confianza = probabilidades[indice].item() * 100

    return categoria, confianza


# ==========================================
# RECONOCIMIENTO DEL CONTENIDO CON RESNET50
# ==========================================

def reconocer_contenido(imagen):
    tensor = transformacion_resnet(imagen).unsqueeze(0)

    with torch.inference_mode():
        salida = modelo_resnet(tensor)
        probabilidades = torch.softmax(salida, dim=1)[0]

    valores, indices = torch.topk(probabilidades, 5)

    predicciones = []

    for valor, indice in zip(valores, indices):
        nombre = categorias_resnet[indice.item()]
        porcentaje = valor.item() * 100

        predicciones.append({
            "nombre": nombre,
            "confianza": porcentaje
        })

    return predicciones


# ==========================================
# MOSTRAR PREDICCIONES DEL CONTENIDO
# ==========================================

def mostrar_contenido(predicciones):
    texto_contenido.config(state="normal")
    texto_contenido.delete("1.0", tk.END)

    if predicciones:
        principal = predicciones[0]

        texto_contenido.insert(
            tk.END,
            "PREDICCIÓN PRINCIPAL\n"
        )

        texto_contenido.insert(
            tk.END,
            f"{principal['nombre'].upper()} "
            f"- {principal['confianza']:.2f}%\n\n"
        )

        texto_contenido.insert(
            tk.END,
            "CINCO PREDICCIONES PRINCIPALES\n"
        )

        for numero, prediccion in enumerate(predicciones, start=1):
            texto_contenido.insert(
                tk.END,
                f"{numero}. {prediccion['nombre'].upper()} "
                f"- {prediccion['confianza']:.2f}%\n"
            )

        texto_contenido.insert(
            tk.END,
            "\nNota: estos porcentajes son puntuaciones del modelo, "
            "no garantías de que la identificación sea correcta."
        )

    else:
        texto_contenido.insert(
            tk.END,
            "No se pudo reconocer el contenido de la imagen."
        )

    texto_contenido.config(state="disabled")


# ==========================================
# ANÁLISIS FORENSE Y EXTRACCIÓN DE METADATOS
# ==========================================

def calcular_sha256(ruta):
    """Calcula el hash SHA-256 del archivo."""

    sha256 = hashlib.sha256()

    with open(ruta, "rb") as archivo:
        for bloque in iter(
            lambda: archivo.read(1024 * 1024), b""
        ):
            sha256.update(bloque)

    return sha256.hexdigest()


def analizar_metadatos(ruta):
    """Extrae información técnica y metadatos disponibles."""

    ruta = Path(ruta)
    datos = []

    estadisticas = ruta.stat()
    tamano_bytes = estadisticas.st_size

    fecha_modificacion = datetime.fromtimestamp(
        estadisticas.st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")

    datos.append("INFORMACIÓN DEL ARCHIVO")
    datos.append(f"Nombre: {ruta.name}")
    datos.append(
        f"Formato de archivo: {ruta.suffix.lower() or 'No identificado'}"
    )
    datos.append(
        f"Tamaño: {tamano_bytes:,} bytes "
        f"({tamano_bytes / 1024:.2f} KB)"
    )
    datos.append(
        f"Última modificación del archivo: {fecha_modificacion}"
    )

    with Image.open(ruta) as imagen:

        datos.append(
            f"Formato detectado: {imagen.format or 'No identificado'}"
        )
        datos.append(
            f"Dimensiones: {imagen.width} x {imagen.height} píxeles"
        )
        datos.append(f"Modo de color: {imagen.mode}")

        exif = imagen.getexif()
        exif_encontrados = {}

        for identificador, valor in exif.items():
            nombre = TAGS.get(identificador, str(identificador))
            exif_encontrados[nombre] = valor

        campos_interes = [
            "Make",
            "Model",
            "Software",
            "DateTime",
            "DateTimeOriginal",
            "DateTimeDigitized",
            "Artist",
            "Copyright",
            "Orientation"
        ]

        datos.append("")
        datos.append("METADATOS EXIF")

        if exif_encontrados:

            datos.append("Se encontraron metadatos EXIF.")

            for campo in campos_interes:
                if campo in exif_encontrados:

                    valor = exif_encontrados[campo]

                    if isinstance(valor, bytes):
                        valor = valor.decode(
                            "utf-8",
                            errors="replace"
                        )

                    datos.append(f"{campo}: {valor}")

            campos_mostrados = [
                campo for campo in campos_interes
                if campo in exif_encontrados
            ]

            if not campos_mostrados:
                datos.append(
                    "Hay información EXIF, pero no contiene "
                    "los campos principales consultados."
                )

            if 34853 in exif:
                datos.append(
                    "Se detectó un bloque de información GPS EXIF. "
                    "Por privacidad, no se muestran coordenadas."
                )

        else:
            datos.append("No se encontraron metadatos EXIF.")

        # Información adicional del archivo
        datos.append("")
        datos.append("OTRA INFORMACIÓN INCORPORADA")

        claves_info = [
            "Software",
            "Comment",
            "Description",
            "Author",
            "Title",
            "Creation Time",
            "XML:com.adobe.xmp"
        ]

        info_mostrada = False

        for clave in claves_info:

            if clave in imagen.info:

                valor = imagen.info[clave]

                if isinstance(valor, bytes):
                    valor = valor.decode(
                        "utf-8",
                        errors="replace"
                    )

                valor = str(valor).replace("\n", " ")

                if len(valor) > 300:
                    valor = valor[:300] + "..."

                datos.append(f"{clave}: {valor}")
                info_mostrada = True

        if not info_mostrada:
            datos.append(
                "No se encontraron campos adicionales conocidos."
            )

    datos.append("")
    datos.append("VERIFICACIÓN DE INTEGRIDAD")
    datos.append(f"SHA-256: {calcular_sha256(ruta)}")

    datos.append("")
    datos.append(
        "OBSERVACIÓN: la presencia, ausencia o modificación de "
        "metadatos no demuestra por sí sola que una imagen sea real "
        "o generada por IA. Los metadatos pueden perderse o modificarse "
        "al editar o compartir un archivo."
    )

    return "\n".join(datos)


def mostrar_informe_forense(informe):

    texto_forense.config(state="normal")
    texto_forense.delete("1.0", tk.END)
    texto_forense.insert(tk.END, informe)
    texto_forense.config(state="disabled")


# ==========================================
# ANALIZAR IMAGEN
# ==========================================

def analizar_imagen():
    global imagen_mostrada

    if not ruta_seleccionada:
        return

    boton_analizar.config(state="disabled")
    ventana.config(cursor="watch")
    ventana.update()

    try:
        imagen = Image.open(
            ruta_seleccionada
        ).convert("RGB")

        # Primera tarea: determinar REAL / FAKE
        categoria, confianza = detectar_origen(imagen)

        # Segunda tarea: reconocer el contenido
        predicciones = reconocer_contenido(imagen)

        # Tercera tarea: análisis forense
        informe_forense = analizar_metadatos(
            ruta_seleccionada
        )

        # Mostrar imagen
        vista = imagen.copy()
        vista.thumbnail((500, 330))

        imagen_mostrada = ImageTk.PhotoImage(vista)

        etiqueta_imagen.config(
            image=imagen_mostrada,
            text=""
        )

        # Mostrar resultado REAL / FAKE
        if categoria.upper() == "FAKE":

            etiqueta_resultado.config(
                text="IMAGEN CLASIFICADA COMO GENERADA POR IA",
                fg="#c0392b"
            )

        else:

            etiqueta_resultado.config(
                text="IMAGEN CLASIFICADA COMO REAL",
                fg="#218c52"
            )

        etiqueta_confianza.config(
            text=f"Puntuación del modelo REAL/FAKE: {confianza:.2f}%"
        )

        # Mostrar resultado ResNet50
        mostrar_contenido(predicciones)

        # Mostrar informe forense
        mostrar_informe_forense(informe_forense)

    except Exception as error:

        messagebox.showerror(
            "Error de análisis",
            str(error)
        )

    finally:

        ventana.config(cursor="")
        boton_analizar.config(state="normal")


# ==========================================
# INTERFAZ GRÁFICA
# ==========================================

ventana = tk.Tk()

ventana.title(
    "Sistema de detección de imágenes generadas por IA"
)

ventana.geometry("1000x850")
ventana.minsize(800, 600)
ventana.configure(bg="#f2f4f7")


# Contenedor con desplazamiento vertical
contenedor_scroll = tk.Frame(
    ventana,
    bg="#f2f4f7"
)

contenedor_scroll.pack(
    fill="both",
    expand=True
)


lienzo = tk.Canvas(
    contenedor_scroll,
    bg="#f2f4f7",
    highlightthickness=0
)

barra_vertical = tk.Scrollbar(
    contenedor_scroll,
    orient="vertical",
    command=lienzo.yview
)

lienzo.configure(
    yscrollcommand=barra_vertical.set
)

barra_vertical.pack(
    side="right",
    fill="y"
)

lienzo.pack(
    side="left",
    fill="both",
    expand=True
)


marco_principal = tk.Frame(
    lienzo,
    bg="#f2f4f7"
)

ventana_contenido = lienzo.create_window(
    (0, 0),
    window=marco_principal,
    anchor="nw"
)


def actualizar_area_desplazamiento(event=None):

    lienzo.configure(
        scrollregion=lienzo.bbox("all")
    )

    lienzo.itemconfigure(
        ventana_contenido,
        width=lienzo.winfo_width()
    )


marco_principal.bind(
    "<Configure>",
    actualizar_area_desplazamiento
)

lienzo.bind(
    "<Configure>",
    actualizar_area_desplazamiento
)


def desplazar_con_rueda(event):

    lienzo.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


lienzo.bind_all(
    "<MouseWheel>",
    desplazar_con_rueda
)


# ==========================================
# TÍTULO
# ==========================================

titulo = tk.Label(
    marco_principal,
    text="DETECTOR DE IMÁGENES CON IA",
    font=("Arial", 22, "bold"),
    bg="#f2f4f7",
    fg="#203864"
)

titulo.pack(pady=(20, 5))


subtitulo = tk.Label(
    marco_principal,
    text="Detección de imágenes sintéticas y reconocimiento de contenido",
    font=("Arial", 10),
    bg="#f2f4f7",
    fg="#555555"
)

subtitulo.pack(pady=(0, 15))


# ==========================================
# VISTA PREVIA DE LA IMAGEN
# ==========================================

marco_imagen = tk.Frame(
    marco_principal,
    bg="white",
    width=550,
    height=350,
    highlightbackground="#cccccc",
    highlightthickness=1
)

marco_imagen.pack(pady=8)
marco_imagen.pack_propagate(False)


etiqueta_imagen = tk.Label(
    marco_imagen,
    text="Selecciona una imagen para comenzar",
    font=("Arial", 13),
    bg="white",
    fg="#777777"
)

etiqueta_imagen.pack(expand=True)


etiqueta_archivo = tk.Label(
    marco_principal,
    text="Ningún archivo seleccionado",
    font=("Arial", 10),
    bg="#f2f4f7",
    fg="#555555"
)

etiqueta_archivo.pack(pady=5)


# ==========================================
# BOTONES
# ==========================================

boton_seleccionar = tk.Button(
    marco_principal,
    text="Seleccionar imagen",
    command=seleccionar_imagen,
    font=("Arial", 11, "bold"),
    bg="#203864",
    fg="white",
    relief="flat",
    padx=20,
    pady=9,
    cursor="hand2"
)

boton_seleccionar.pack(pady=5)


boton_analizar = tk.Button(
    marco_principal,
    text="Analizar imagen",
    command=analizar_imagen,
    font=("Arial", 11, "bold"),
    bg="#218c52",
    fg="white",
    relief="flat",
    padx=25,
    pady=9,
    cursor="hand2",
    state="disabled"
)

boton_analizar.pack(pady=5)


# ==========================================
# RESULTADO REAL / FAKE
# ==========================================

etiqueta_resultado = tk.Label(
    marco_principal,
    text="Esperando imagen...",
    font=("Arial", 15, "bold"),
    bg="#f2f4f7",
    fg="#34495e",
    wraplength=850
)

etiqueta_resultado.pack(pady=(12, 3))


etiqueta_confianza = tk.Label(
    marco_principal,
    text="",
    font=("Arial", 10),
    bg="#f2f4f7",
    fg="#555555"
)

etiqueta_confianza.pack(pady=3)


# ==========================================
# PANEL DE RECONOCIMIENTO
# ==========================================

titulo_contenido = tk.Label(
    marco_principal,
    text="RECONOCIMIENTO DEL CONTENIDO DE LA IMAGEN",
    font=("Arial", 12, "bold"),
    bg="#f2f4f7",
    fg="#203864"
)

titulo_contenido.pack(pady=(12, 4))


texto_contenido = tk.Text(
    marco_principal,
    height=9,
    width=75,
    font=("Arial", 10),
    bg="white",
    fg="#333333",
    relief="solid",
    borderwidth=1,
    wrap="word"
)

texto_contenido.pack(
    pady=5,
    padx=20,
    fill="x"
)

texto_contenido.insert(
    tk.END,
    "Las predicciones del contenido aparecerán aquí después del análisis."
)

texto_contenido.config(state="disabled")


# ==========================================
# PANEL DE ANÁLISIS FORENSE
# ==========================================

titulo_forense = tk.Label(
    marco_principal,
    text="ANÁLISIS FORENSE Y METADATOS",
    font=("Arial", 12, "bold"),
    bg="#f2f4f7",
    fg="#203864"
)

titulo_forense.pack(pady=(14, 4))


texto_forense = tk.Text(
    marco_principal,
    height=16,
    width=75,
    font=("Consolas", 9),
    bg="white",
    fg="#333333",
    relief="solid",
    borderwidth=1,
    wrap="word"
)

texto_forense.pack(
    pady=5,
    padx=20,
    fill="both",
    expand=True
)

texto_forense.insert(
    tk.END,
    "El informe forense aparecerá aquí después del análisis."
)

texto_forense.config(state="disabled")


# ==========================================
# INICIO
# ==========================================

try:

    etiqueta_resultado.config(
        text="Cargando modelos...",
        fg="#34495e"
    )

    ventana.update()

    cargar_modelos()

    etiqueta_resultado.config(
        text="Modelos cargados. Esperando imagen...",
        fg="#218c52"
    )

    ventana.mainloop()

except Exception as error:

    messagebox.showerror(
        "Error al iniciar el sistema",
        str(error),
        parent=ventana
    )

    ventana.destroy()
