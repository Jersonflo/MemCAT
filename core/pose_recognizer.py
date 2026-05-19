"""
MemCAT - Reconocedor de poses gestuales.
Analiza landmarks de manos y determina si se hace la "CatPose":
  - Una mano alzada con palma abierta (la mano del baile 🙌)
  - La otra mano empuñada cerca del rostro/boca (el micrófono 🎤)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from core.hand_detector import HandLandmarks, HandIdx


# ─── Tipos de resultado ────────────────────────────────────────────────────────

@dataclass
class GestureState:
    """
    Estado actual de los gestos detectados.

    Attributes:
        has_raised_open:  True si hay al menos una mano alzada y abierta.
        has_fist:         True si hay al menos una mano en puño.
        cat_pose_active:  True si AMBOS gestos están presentes → activar GIF.
        hold_frames:      Cuántos frames consecutivos lleva activa la pose.
        debug_info:       Diccionario con valores de diagnóstico.
    """
    has_raised_open: bool = False
    has_fist: bool = False
    cat_pose_active: bool = False
    hold_frames: int = 0
    debug_info: dict = field(default_factory=dict)


# ─── Reconocedor principal ─────────────────────────────────────────────────────

class PoseRecognizer:
    """
    Analiza la lista de HandLandmarks y determina si se está haciendo la CatPose.

    Lógica de detección:
      1. Mano alzada + abierta:
         - wrist.y < RAISED_THRESHOLD  (posición alta en pantalla)
         - ≥ 3 dedos extendidos (fingertip por encima de su PIP)
      2. Mano en puño:
         - ≥ FIST_MIN_CURLED dedos doblados (fingertip por debajo de su MCP)
         - wrist.y > FIST_MAX_Y (no tan arriba, está cerca del rostro)

    La pose se activa cuando ambas condiciones se cumplen al mismo tiempo
    durante al menos HOLD_FRAMES frames consecutivos.
    """

    def __init__(
        self,
        raised_y_threshold: float = 0.42,
        fist_min_curled: int = 3,
        fist_max_y: float = 0.80,
        hold_frames: int = 8,
    ) -> None:
        """
        Args:
            raised_y_threshold: Y normalizado máximo para considerar mano alzada
                                 (menor y = más arriba en la imagen).
            fist_min_curled:    Dedos mínimos doblados para ser puño.
            fist_max_y:         Y normalizado máximo de la muñeca del puño
                                 (evita contar manos muy abajo).
            hold_frames:        Frames consecutivos necesarios para activar.
        """
        self._raised_y = raised_y_threshold
        self._fist_min = fist_min_curled
        self._fist_max_y = fist_max_y
        self._hold_frames = hold_frames
        self._consecutive: int = 0

    # ─── API pública ──────────────────────────────────────────────────────────

    def evaluate(self, hands: list[HandLandmarks]) -> GestureState:
        """
        Evalúa la lista de manos detectadas y devuelve el estado de gestos.

        Args:
            hands: Lista de HandLandmarks del frame actual.

        Returns:
            GestureState con el resultado de la evaluación.
        """
        state = GestureState()

        if not hands:
            self._consecutive = 0
            return state

        raised_open_found = False
        fist_found = False
        debug: dict = {}

        for hand in hands:
            lms = hand.landmarks_n  # list[(x, y, z)]
            wrist_y = lms[HandIdx.WRIST][1]

            curled = self._count_curled_fingers(lms)
            extended = self._count_extended_fingers(lms)

            debug[hand.handedness] = {
                "wrist_y": round(wrist_y, 3),
                "curled": curled,
                "extended": extended,
            }

            # ── Mano arriba (Cualquier punto arriba de la mitad) ───────────
            # Detecta si cualquier punto de la mano está en la mitad superior (y < 0.5)
            if any(lm[1] < 0.5 for lm in lms):
                raised_open_found = True

            # ── Puño ──────────────────────────────────────────────────────
            if curled >= self._fist_min and wrist_y < self._fist_max_y:
                fist_found = True

        state.has_raised_open = raised_open_found
        state.has_fist = fist_found
        state.debug_info = debug

        # Incrementar / reiniciar contador de frames consecutivos
        if raised_open_found and fist_found:
            self._consecutive += 1
        else:
            self._consecutive = max(0, self._consecutive - 1)

        state.hold_frames = self._consecutive
        state.cat_pose_active = self._consecutive >= self._hold_frames

        return state

    # ─── Helpers privados ─────────────────────────────────────────────────────

    @staticmethod
    def _count_curled_fingers(lms: list[tuple[float, float, float]]) -> int:
        """
        Cuenta cuántos dedos (sin pulgar) están doblados.
        Un dedo está doblado si su punta está POR DEBAJO de su articulación MCP
        (y mayor = más abajo en la imagen).
        """
        curled = 0
        for mcp_idx, pip_idx, tip_idx in HandIdx.FINGER_TRIPLETS:
            tip_y = lms[tip_idx][1]
            mcp_y = lms[mcp_idx][1]
            if tip_y > mcp_y:  # punta más baja que la base = doblado
                curled += 1
        return curled

    @staticmethod
    def _count_extended_fingers(lms: list[tuple[float, float, float]]) -> int:
        """
        Cuenta cuántos dedos (sin pulgar) están extendidos.
        Un dedo está extendido si su punta está POR ENCIMA de su articulación PIP.
        """
        extended = 0
        for mcp_idx, pip_idx, tip_idx in HandIdx.FINGER_TRIPLETS:
            tip_y = lms[tip_idx][1]
            pip_y = lms[pip_idx][1]
            if tip_y < pip_y:  # punta más alta que PIP = extendido
                extended += 1
        return extended
