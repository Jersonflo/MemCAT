# MemCAT 🐱

> Proyecto de entretenimiento con **Visión por Computadora** usando Python + MediaPipe.
> Haz la **CatPose** y aparece el gatito bailando.

---

## 🎯 ¿Qué hace?

Detecta una pose específica con ambas manos usando tu webcam:

| Mano | Gesto | |
|---|---|---|
| Cualquier mano | ✋ Alzada + abierta | Dedos extendidos hacia arriba |
| Otra mano | ✊ Puño | Dedos cerrados |

Cuando ambos gestos se mantienen juntos por ~8 frames → **¡aparece el gatito bailando!** 🐈

---

## 🚀 Instalación rápida

### 1. Crear entorno virtual (recomendado)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Descargar el GIF del gatito
```bash
python setup_assets.py
```

### 4. ¡Ejecutar!
```bash
python main.py
```

---

## 📁 Estructura del proyecto

```
MemCAT/
├── config/
│   └── settings.py         # ⚙ Todos los parámetros ajustables aquí
├── core/
│   ├── capture.py          # 📷 Captura de video multi-hilo
│   ├── hand_detector.py    # 🤚 Detección con MediaPipe Hands
│   ├── pose_recognizer.py  # 🧠 Lógica de reconocimiento de CatPose
│   └── gif_overlay.py      # 🐱 Overlay del GIF animado con alpha
├── pipeline/
│   └── pipeline.py         # 🎬 Orquestador principal
├── utils/
│   └── fps_counter.py      # ⏱ Contador de FPS estable
├── assets/
│   └── cat.gif             # (descargado por setup_assets.py)
├── main.py                 # 🚀 Punto de entrada
├── setup_assets.py         # 📥 Descargador del GIF
└── requirements.txt
```

---

## ⚙ Ajustes en `config/settings.py`

| Parámetro | Default | Descripción |
|---|---|---|
| `RAISED_HAND_Y_THRESHOLD` | `0.42` | Qué tan arriba debe estar la mano alzada |
| `FIST_MIN_CURLED_FINGERS` | `3` | Dedos mínimos cerrados para ser puño |
| `POSE_HOLD_FRAMES` | `8` | Frames que debe mantenerse la pose |
| `GIF_SIZE` | `(280, 280)` | Tamaño del GIF en pantalla |
| `GIF_POSITION` | `"top-right"` | Posición: top-left/right, bottom-left/right, center |
| `GIF_SPEED` | `2` | Velocidad de animación del GIF |

---

## 🎮 Controles

| Tecla | Acción |
|---|---|
| `q` / `ESC` | Salir |

---

## 🛠 Si el GIF no se descarga automáticamente

Coloca cualquier GIF de un gato bailando en:
```
assets/cat.gif
```
¡Cualquier GIF funciona! 🐱
