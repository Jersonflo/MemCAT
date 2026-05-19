"""
MemCAT - Script de setup de assets.
Descarga el GIF del gatito bailando desde internet.
Ejecutar una vez antes de lanzar main.py.
"""
import sys
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Forzar stdout en UTF-8 en Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import urllib.request
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"

# URLs candidatas del GIF del gatito (se intenta en orden)
CAT_GIF_URLS: list[str] = [
    "https://media.tenor.com/7MZcKWIiGiQAAAAd/cat-dance.gif",
    "https://media.tenor.com/NeNkVHHNLokAAAAd/cat.gif",
    "https://media.giphy.com/media/ICOgUNjpvO0PC/giphy.gif",
    "https://media.giphy.com/media/8vQSQ3cNXuDGo/giphy.gif",
]

GIF_PATH = ASSETS_DIR / "cat.gif"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = ASSETS_DIR / "hand_landmarker.task"


def download_gif() -> bool:
    """
    Intenta descargar el GIF del gatito desde las URLs configuradas.

    Returns:
        True si se descargó correctamente, False si todos los intentos fallaron.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(CAT_GIF_URLS, 1):
        print(f"  [{i}/{len(CAT_GIF_URLS)}] Intentando: {url[:60]}...")
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                )
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read()

            if not data.startswith(b"GIF"):
                print(f"       [!] Respuesta no es un GIF valido ({len(data)} bytes).")
                continue

            GIF_PATH.write_bytes(data)
            print(f"       [OK] Descargado! ({len(data) / 1024:.1f} KB) -> {GIF_PATH}")
            return True

        except Exception as e:
            print(f"       [X] Error: {e}")

    return False


def download_model() -> bool:
    """Descarga el modelo de MediaPipe Hand Landmarker."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    if MODEL_PATH.exists():
        print(f"  [OK] El modelo ya existe: {MODEL_PATH.name}")
        return True
        
    print(f"  [>>] Descargando modelo...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(MODEL_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            
        MODEL_PATH.write_bytes(data)
        print(f"       [OK] Modelo descargado! ({len(data) / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"       [X] Error descargando modelo: {e}")
        return False


def main() -> None:
    print("=" * 55)
    print("  MemCAT - Setup de Assets")
    print("=" * 55)

    # 1. Descargar GIF
    if GIF_PATH.exists():
        size_kb = GIF_PATH.stat().st_size / 1024
        print(f"\n[OK] El GIF ya existe ({size_kb:.1f} KB): {GIF_PATH}")
        ans = input("  Descargar de nuevo? [s/N]: ").strip().lower()
        if ans == "s":
            print("[>>] Descargando GIF del gatito bailando...")
            download_gif()
    else:
        print("\n[>>] Descargando GIF del gatito bailando...")
        download_gif()

    # 2. Descargar Modelo
    print("\n[>>] Verificando modelo de MediaPipe...")
    download_model()

    print("\n[OK] Setup completado! Ahora ejecuta: python main.py")


if __name__ == "__main__":
    main()
