"""
MemCAT - Módulo de captura de video.
Wrapper robusto sobre cv2.VideoCapture con context manager.
"""
import threading
import queue
import cv2
import numpy as np
from typing import Generator


class VideoCapture:
    """
    Captura de video robusta sobre cv2.VideoCapture.
    Soporta webcam, archivos de video y streams RTSP/IP.
    Usa threading interno para evitar bloqueos de I/O en el hilo principal.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        buffer_size: int = 2,
    ) -> None:
        """
        Args:
            source:      Índice de webcam (int) o ruta/URL de video (str).
            width:       Ancho del frame en píxeles.
            height:      Alto del frame en píxeles.
            buffer_size: Máximo de frames en cola (menor = menor latencia).
        """
        self._source = source
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._is_open: bool = False

        # Buffer producer-consumer para captura en hilo separado
        self._buffer: queue.Queue[np.ndarray] = queue.Queue(maxsize=buffer_size)
        self._running: bool = False
        self._thread: threading.Thread | None = None

    # ─── Propiedades públicas ─────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """True si la captura está activa."""
        return self._is_open

    @property
    def fps(self) -> float:
        """FPS reportados por la fuente de video."""
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0

    # ─── Ciclo de vida ────────────────────────────────────────────────────────

    def open(self) -> None:
        """Abre la fuente de video y lanza el hilo de captura."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[VideoCapture] No se pudo abrir la fuente: {self._source}\n"
                "Verifica que la webcam esté conectada o que la ruta sea válida."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._is_open = True
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def release(self) -> None:
        """Detiene el hilo de captura y libera el dispositivo."""
        self._running = False
        self._is_open = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    # ─── Lectura de frames ────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """
        Lee el frame más reciente del buffer.

        Returns:
            Frame BGR como numpy array, o None si no hay disponible.
        """
        try:
            return self._buffer.get(timeout=0.5)
        except queue.Empty:
            return None

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generador que produce frames en orden mientras la captura esté activa.

        Yields:
            Frame BGR como numpy array.
        """
        while self._running:
            frame = self.read()
            if frame is not None:
                yield frame

    # ─── Hilo interno ─────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Bucle de captura en hilo daemon. Llena el buffer con frames."""
        while self._running and self._cap:
            ret, frame = self._cap.read()
            if not ret:
                self._running = False
                break
            # Descartar frame antiguo si el buffer está lleno (priorizar frescura)
            if self._buffer.full():
                try:
                    self._buffer.get_nowait()
                except queue.Empty:
                    pass
            self._buffer.put(frame)

    # ─── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "VideoCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
