"""
MemCAT - Detector de manos con MediaPipe (Tasks API).
Encapsula MediaPipe HandLandmarker y expone landmarks normalizados y en píxeles.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# ─── Tipos de datos ────────────────────────────────────────────────────────────

@dataclass
class HandLandmarks:
    """
    Contenedor de landmarks de una mano detectada.

    Attributes:
        handedness:   'Left' o 'Right' (desde la perspectiva del usuario).
        score:        Confianza de la detección [0-1].
        landmarks_n:  Lista de 21 puntos (x, y, z) normalizados [0-1].
        landmarks_px: Lista de 21 puntos (x, y) en píxeles absolutos.
    """
    handedness: str
    score: float
    landmarks_n: list[tuple[float, float, float]]   # normalizados
    landmarks_px: list[tuple[int, int]]              # píxeles


# Índices de referencia de MediaPipe Hands
class HandIdx:
    """Índices de landmarks según el modelo MediaPipe Hands."""
    WRIST = 0
    THUMB_TIP = 4
    INDEX_MCP = 5;  INDEX_PIP = 6;  INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_TIP = 12
    RING_MCP = 13;  RING_PIP = 14;  RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_TIP = 20

    # Tuplas (MCP, PIP, TIP) de cada dedo (sin pulgar)
    FINGER_TRIPLETS: list[tuple[int, int, int]] = [
        (INDEX_MCP,  INDEX_PIP,  INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP,   RING_PIP,   RING_TIP),
        (PINKY_MCP,  PINKY_PIP,  PINKY_TIP),
    ]


# ─── Clase principal ───────────────────────────────────────────────────────────

class HandDetector:
    """
    Wrapper de MediaPipe Hands para detección y extracción de landmarks.
    Usa la nueva Tasks API.
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.75,
        tracking_confidence: float = 0.60,
    ) -> None:
        """
        Args:
            max_hands:             Número máximo de manos a detectar.
            detection_confidence:  Confianza mínima para detección inicial.
            tracking_confidence:   Confianza mínima para tracking continuo.
        """
        self._max_hands = max_hands
        self._det_conf = detection_confidence
        self._trk_conf = tracking_confidence

        # El modelo se descarga en assets/ por setup_assets.py
        self._model_path = str(Path(__file__).parent.parent / "assets" / "hand_landmarker.task")
        self._detector: vision.HandLandmarker | None = None

    # ─── Ciclo de vida ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Inicializa el modelo MediaPipe HandLandmarker."""
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"No se encontro el modelo en {self._model_path}. "
                "Ejecuta primero: python setup_assets.py"
            )
            
        base_options = python.BaseOptions(model_asset_path=self._model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._det_conf,
            min_hand_presence_confidence=self._trk_conf,
            running_mode=vision.RunningMode.IMAGE
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        print("[HandDetector] Modelo MediaPipe HandLandmarker cargado [OK]")

    def release(self) -> None:
        """Libera los recursos del modelo."""
        if self._detector:
            self._detector.close()
            self._detector = None

    # ─── Detección ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> list[HandLandmarks]:
        """
        Detecta manos en un frame BGR y retorna sus landmarks.

        Args:
            frame_bgr: Frame en formato BGR (salida de OpenCV).

        Returns:
            Lista de HandLandmarks; vacía si no hay manos detectadas.
        """
        if self._detector is None:
            raise RuntimeError("Llama a load() antes de detect().")

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Convertir a mediapipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detectar
        result = self._detector.detect(mp_image)
        
        detected: list[HandLandmarks] = []
        
        if not result.hand_landmarks:
            return detected

        for hand_lms, handedness in zip(result.hand_landmarks, result.handedness):
            # handedness es una lista de categorías
            label = handedness[0].category_name
            score = handedness[0].score
            
            lms_n: list[tuple[float, float, float]] = []
            lms_px: list[tuple[int, int]] = []
            
            for lm in hand_lms:
                lms_n.append((lm.x, lm.y, lm.z))
                lms_px.append((int(lm.x * w), int(lm.y * h)))
                
            detected.append(HandLandmarks(
                handedness=label,
                score=score,
                landmarks_n=lms_n,
                landmarks_px=lms_px,
            ))
            
        return detected

    # ─── Dibujo ───────────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        hands: list[HandLandmarks],
        color: tuple[int, int, int] = (0, 255, 180),
    ) -> np.ndarray:
        """
        Dibuja landmarks y conexiones sobre el frame.

        Args:
            frame:  Frame BGR destino.
            hands:  Lista de HandLandmarks a dibujar.
            color:  Color de los puntos en BGR.

        Returns:
            Frame con los landmarks dibujados.
        """
        if not hands:
            return frame

        # Conexiones manuales
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        for hand in hands:
            # Dibujar conexiones
            for start_idx, end_idx in connections:
                if start_idx < len(hand.landmarks_px) and end_idx < len(hand.landmarks_px):
                    pt1 = hand.landmarks_px[start_idx]
                    pt2 = hand.landmarks_px[end_idx]
                    cv2.line(frame, pt1, pt2, (color[0]//2, color[1]//2, color[2]//2), 2)
                
            # Dibujar puntos
            for pt in hand.landmarks_px:
                cv2.circle(frame, pt, 5, color, -1)

            # Etiqueta de handedness
            wrist_px = hand.landmarks_px[HandIdx.WRIST]
            label_text = f"{hand.handedness} ({hand.score:.0%})"
            cv2.putText(
                frame, label_text,
                (wrist_px[0] - 30, wrist_px[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        return frame

    # ─── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "HandDetector":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.release()
