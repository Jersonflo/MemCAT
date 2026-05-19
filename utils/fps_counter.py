"""
MemCAT - Contador de FPS con ventana deslizante.
Provee una medición estable evitando picos de un solo frame.
"""
import time
from collections import deque


class FPSCounter:
    """
    Contador de FPS basado en ventana de tiempo deslizante.
    Más estable que medir solo el delta del último frame.
    """

    def __init__(self, window_size: int = 30) -> None:
        """
        Args:
            window_size: Número de frames a considerar para el promedio.
        """
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """
        Registra el timestamp actual y retorna el FPS promedio.

        Returns:
            FPS promedio en la ventana. 0.0 si no hay suficientes muestras.
        """
        self._timestamps.append(time.perf_counter())
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def reset(self) -> None:
        """Limpia el historial de timestamps."""
        self._timestamps.clear()
