"""
MemCAT 🐱
=========
Proyecto de entretenimiento con Visión por Computadora.

Haz la CatPose para que aparezca el gatito bailando:
  ✋ Alza una mano abierta
  ✊ Pon la otra mano en puño

Controles:
  q / ESC → Salir
  d       → Activar/desactivar debug info
  f       → Activar/desactivar pantalla completa
"""
import sys
from pathlib import Path

# Asegurar que el directorio raíz esté en el path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.settings import Config
from pipeline.pipeline import MemCATpipeline


def main() -> None:
    """Punto de entrada de MemCAT."""
    # ── Verificar que el GIF existe ───────────────────────────────────────────
    gif_path = ROOT / "assets" / "cat.gif"
    if not gif_path.exists():
        print("\n[!] El GIF del gatito no esta descargado.")
        print("   Ejecuta primero: python setup_assets.py\n")
        ans = input("  Intentar descargarlo ahora? [S/n]: ").strip().lower()
        if ans != "n":
            from setup_assets import download_gif
            if not download_gif():
                print("\n[X] No se pudo descargar. Coloca un GIF manualmente en assets/cat.gif")
                sys.exit(1)
        else:
            print("\n  Continuando sin GIF (la pose seguira detectandose).")

    # ── Configuración ─────────────────────────────────────────────────────────
    config = Config(
        SOURCE=0,                         # Webcam principal
        WIDTH=1280,
        HEIGHT=720,
        SHOW_FPS=True,
        SHOW_LANDMARKS=True,
        SHOW_DEBUG_INFO=True,
        GIF_PATH=str(gif_path),
        GIF_POSITION="top-right",
        GIF_SIZE=(280, 280),
        GIF_SPEED=2,
        POSE_HOLD_FRAMES=8,
        RAISED_HAND_Y_THRESHOLD=0.42,
        FIST_MIN_CURLED_FINGERS=3,
    )

    # ── Lanzar pipeline ───────────────────────────────────────────────────────
    pipeline = MemCATpipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
