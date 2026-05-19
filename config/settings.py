"""
MemCAT - Configuración centralizada del proyecto.
Todos los parámetros ajustables del sistema viven aquí.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Clase de configuración principal de MemCAT.
    Centraliza todos los parámetros del pipeline CV.
    """

    # ─── Fuente de video ─────────────────────────────────────────────────────
    SOURCE: int | str = 0         # 0 = webcam principal
    WIDTH: int = 1280
    HEIGHT: int = 720
    TARGET_FPS: int = 30

    # ─── MediaPipe Hands ─────────────────────────────────────────────────────
    MAX_HANDS: int = 2
    DETECTION_CONFIDENCE: float = 0.40
    TRACKING_CONFIDENCE: float = 0.40

    # ─── Detección de gestos ─────────────────────────────────────────────────
    # Umbral Y normalizado [0-1]: mano alzada si wrist.y < este valor
    RAISED_HAND_Y_THRESHOLD: float = 0.85

    # Ratio fingertip-vs-MCP para decidir si un dedo está doblado
    # (menor ratio = más cerrado)
    FIST_CURL_RATIO: float = 0.85

    # Número mínimo de dedos doblados para considerar que es puño
    FIST_MIN_CURLED_FINGERS: int = 2

    # Cuántos frames consecutivos debe mantenerse la pose para activar el GIF
    POSE_HOLD_FRAMES: int = 15

    # ─── GIF overlay ─────────────────────────────────────────────────────────
    GIF_PATH: str = "assets/cat.gif"
    GIF_SIZE: tuple[int, int] = field(default_factory=lambda: (280, 280))
    GIF_POSITION: str = "top-right"   # "top-left" | "top-right" | "center"
    GIF_SPEED: int = 2                # frames de video por frame de GIF

    # ─── Efectos visuales ─────────────────────────────────────────────────────
    # Color de landmarks cuando la pose NO está activa (BGR)
    LANDMARK_COLOR_IDLE: tuple[int, int, int] = field(
        default_factory=lambda: (200, 200, 200)
    )
    # Color cuando la pose SÍ está activa
    LANDMARK_COLOR_ACTIVE: tuple[int, int, int] = field(
        default_factory=lambda: (0, 255, 180)
    )

    # ─── Display ──────────────────────────────────────────────────────────────
    SHOW_FPS: bool = True
    SHOW_LANDMARKS: bool = True
    SHOW_DEBUG_INFO: bool = True
    WINDOW_NAME: str = "MemCAT 🐱"

    # ─── Setup de assets ──────────────────────────────────────────────────────
    CAT_GIF_URLS: list[str] = field(default_factory=lambda: [
        "https://media.tenor.com/7MZcKWIiGiQAAAAM/cat-dance.gif",
        "https://media.tenor.com/NeNkVHHNLokAAAAM/cat.gif",
        "https://media.giphy.com/media/ICOgUNjpvO0PC/giphy.gif",
    ])
