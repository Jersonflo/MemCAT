"""
MemCAT - Overlay de GIF animado sobre frames de OpenCV.
Carga un GIF, extrae sus frames (con transparencia alpha) y los compone
sobre el frame de video cuando la pose CatPose está activa.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image


class GifOverlay:
    """
    Gestiona la carga, animación y composición de un GIF sobre un frame BGR.

    La composición respeta el canal alpha del GIF (si lo tiene), haciendo
    el fondo transparente sobre el video de forma natural.
    """

    def __init__(
        self,
        gif_path: str,
        size: tuple[int, int] = (280, 280),
        position: str = "top-right",
        speed: int = 2,
    ) -> None:
        """
        Args:
            gif_path: Ruta al archivo .gif.
            size:     (ancho, alto) al que escalar el GIF en píxeles.
            position: Esquina de colocación:
                      "top-left" | "top-right" | "bottom-left" |
                      "bottom-right" | "center"
            speed:    Frames de video por cada frame del GIF
                      (2 = avanza cada 2 frames de video).
        """
        self._gif_path = Path(gif_path)
        self._size = size
        self._position = position
        self._speed = max(1, speed)

        self._frames: list[np.ndarray] = []   # BGRA uint8
        self._frame_idx: int = 0
        self._video_frame_count: int = 0
        self._loaded: bool = False

    # ─── Ciclo de vida ────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Carga el GIF y extrae todos sus frames como arrays BGRA.

        Returns:
            True si la carga fue exitosa, False en caso de error.
        """
        if not self._gif_path.exists():
            print(f"[GifOverlay] ⚠ GIF no encontrado: {self._gif_path}")
            print("  → Ejecuta 'python setup_assets.py' para descargarlo.")
            return False

        try:
            gif = Image.open(self._gif_path)
            self._frames = []
            w, h = self._size

            for frame_idx in range(getattr(gif, "n_frames", 1)):
                gif.seek(frame_idx)
                # Convertir a RGBA para mantener transparencia
                frame_rgba = gif.convert("RGBA").resize((w, h), Image.LANCZOS)
                arr = np.array(frame_rgba, dtype=np.uint8)
                # OpenCV usa BGRA: invertir canales R y B
                bgra = arr[..., [2, 1, 0, 3]]
                self._frames.append(bgra)

            self._loaded = len(self._frames) > 0
            if self._loaded:
                print(f"[GifOverlay] ✔ {len(self._frames)} frames cargados desde {self._gif_path.name}")
            return self._loaded

        except Exception as e:
            print(f"[GifOverlay] Error cargando GIF: {e}")
            return False

    # ─── Composición ──────────────────────────────────────────────────────────

    def apply(
        self,
        frame: np.ndarray,
        active: bool,
    ) -> np.ndarray:
        """
        Compone el GIF sobre el frame si active=True.

        Args:
            frame:  Frame BGR del video (modificado in-place).
            active: Si True, muestra y avanza la animación del GIF.

        Returns:
            Frame con (o sin) el GIF compuesto.
        """
        if not active or not self._loaded or not self._frames:
            # Reiniciar animación al desactivar para que empiece desde el inicio
            self._frame_idx = 0
            self._video_frame_count = 0
            return frame

        # Avanzar frame de GIF según el parámetro de velocidad
        self._video_frame_count += 1
        if self._video_frame_count >= self._speed:
            self._video_frame_count = 0
            self._frame_idx = (self._frame_idx + 1) % len(self._frames)

        gif_bgra = self._frames[self._frame_idx]
        fh, fw = frame.shape[:2]
        gh, gw = gif_bgra.shape[:2]

        # Calcular posición de colocación
        x, y = self._compute_position(fw, fh, gw, gh)

        # Ajustar si el GIF se sale del borde del frame
        x = max(0, min(x, fw - gw))
        y = max(0, min(y, fh - gh))

        # Región de interés en el frame de video
        roi = frame[y : y + gh, x : x + gw]
        if roi.shape[:2] != (gh, gw):
            return frame  # Seguridad: evitar excepciones por bordes

        # Composición alpha blending
        alpha = gif_bgra[..., 3:4].astype(np.float32) / 255.0
        bgr_gif = gif_bgra[..., :3].astype(np.float32)
        bgr_roi = roi.astype(np.float32)

        blended = bgr_gif * alpha + bgr_roi * (1.0 - alpha)
        frame[y : y + gh, x : x + gw] = blended.astype(np.uint8)

        return frame

    # ─── Helpers privados ─────────────────────────────────────────────────────

    def _compute_position(
        self, fw: int, fh: int, gw: int, gh: int
    ) -> tuple[int, int]:
        """Calcula (x, y) de la esquina superior-izquierda del GIF."""
        margin = 20
        positions = {
            "top-left":     (margin, margin),
            "top-right":    (fw - gw - margin, margin),
            "bottom-left":  (margin, fh - gh - margin),
            "bottom-right": (fw - gw - margin, fh - gh - margin),
            "center":       ((fw - gw) // 2, (fh - gh) // 2),
        }
        return positions.get(self._position, (margin, margin))

    # ─── Propiedades ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True si el GIF fue cargado correctamente."""
        return self._loaded

    @property
    def total_frames(self) -> int:
        """Número de frames del GIF."""
        return len(self._frames)
