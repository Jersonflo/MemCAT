"""
MemCAT - Pipeline principal de visión por computadora.
Orquesta: Captura → Detección de manos → Reconocimiento de pose →
          Overlay de GIF → Visualización.
"""
from __future__ import annotations

import cv2
import numpy as np

from config.settings import Config
from core.capture import VideoCapture
from core.hand_detector import HandDetector, HandLandmarks
from core.pose_recognizer import PoseRecognizer, GestureState
from core.gif_overlay import GifOverlay
from utils.fps_counter import FPSCounter


class MemCATpipeline:
    """
    Pipeline principal de MemCAT.

    Ciclo de ejecución por frame:
      1. Leer frame de webcam (hilo productor).
      2. Voltear horizontalmente (selfie view).
      3. Detectar manos con MediaPipe.
      4. Evaluar si se hace la CatPose.
      5. Aplicar GIF overlay si la pose está activa.
      6. Dibujar landmarks + HUD (FPS, estado, debug).
      7. Mostrar en ventana.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._capture = VideoCapture(
            source=config.SOURCE,
            width=config.WIDTH,
            height=config.HEIGHT,
        )
        self._detector = HandDetector(
            max_hands=config.MAX_HANDS,
            detection_confidence=config.DETECTION_CONFIDENCE,
            tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        self._recognizer = PoseRecognizer(
            raised_y_threshold=config.RAISED_HAND_Y_THRESHOLD,
            fist_min_curled=config.FIST_MIN_CURLED_FINGERS,
            hold_frames=config.POSE_HOLD_FRAMES,
        )
        self._gif = GifOverlay(
            gif_path=config.GIF_PATH,
            size=config.GIF_SIZE,
            position=config.GIF_POSITION,
            speed=config.GIF_SPEED,
        )
        self._fps = FPSCounter(window_size=30)

    # ─── API pública ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """Lanza el pipeline. Bloquea hasta que el usuario presione 'q' o ESC."""
        print("\n[MemCAT] Iniciado!")
        print("  --> Alza una mano abierta Y haz un punio con la otra para activar el gatito.")
        print("  --> Presiona 'q' o ESC para salir.\n")

        self._detector.load()
        gif_ok = self._gif.load()
        if not gif_ok:
            print("  [!] El GIF no se cargo. El proyecto funciona sin el.")

        # Crear ventana redimensionable y de tamaño contenido
        cv2.namedWindow(self._cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._cfg.WINDOW_NAME, 800, 600)

        with self._capture:
            for raw_frame in self._capture.frames():
                # 1. Selfie flip (espejo)
                frame = cv2.flip(raw_frame, 1)

                # 2. Detectar manos
                hands = self._detector.detect(frame)

                # 3. Evaluar pose
                state = self._recognizer.evaluate(hands)

                # 4. Overlay del GIF
                # Nota: el GIF se dibuja ANTES de los landmarks para que quede detrás
                frame = self._gif.apply(frame, active=state.cat_pose_active)

                # 5. Dibujar landmarks con color reactivo
                if self._cfg.SHOW_LANDMARKS and hands:
                    color = (
                        self._cfg.LANDMARK_COLOR_ACTIVE
                        if state.cat_pose_active
                        else self._cfg.LANDMARK_COLOR_IDLE
                    )
                    frame = self._detector.draw(frame, hands, color=color)

                # 6. HUD overlay
                frame = self._draw_hud(frame, state)

                # 7. Mostrar
                cv2.imshow(self._cfg.WINDOW_NAME, frame)

                # Verificar si se cerro la ventana con la 'X'
                if cv2.getWindowProperty(self._cfg.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' o ESC
                    break

        cv2.destroyAllWindows()
        self._detector.release()
        print("\n[MemCAT] Finalizado. Hasta la proxima!")

    # ─── Dibujo del HUD ───────────────────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray, state: GestureState) -> np.ndarray:
        """Dibuja FPS, estado de pose y debug info sobre el frame."""
        h, w = frame.shape[:2]
        fps = self._fps.tick()

        # Panel semi-transparente en la esquina inferior izquierda
        overlay = frame.copy()
        panel_h = 130 if self._cfg.SHOW_DEBUG_INFO else 70
        cv2.rectangle(overlay, (0, h - panel_h), (320, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        y_base = h - panel_h + 22

        # FPS
        if self._cfg.SHOW_FPS:
            cv2.putText(
                frame, f"FPS: {fps:.0f}",
                (10, y_base),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2,
            )

        # Estado de la pose
        y_base += 26
        if state.cat_pose_active:
            status_text = "CATPOSE ACTIVADA!"
            status_color = (0, 255, 180)
        elif state.has_raised_open and not state.has_fist:
            status_text = "Mano alzada... falta el punio"
            status_color = (0, 200, 255)
        elif state.has_fist and not state.has_raised_open:
            status_text = "Punio detectado... falta mano alzada"
            status_color = (0, 200, 255)
        else:
            status_text = "Haz la CatPose para el gatito!"
            status_color = (180, 180, 180)

        cv2.putText(
            frame, status_text,
            (10, y_base),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2,
        )

        # Barra de progreso de hold_frames
        if state.has_raised_open and state.has_fist:
            y_base += 22
            progress = min(state.hold_frames / self._cfg.POSE_HOLD_FRAMES, 1.0)
            bar_w = 280
            filled = int(bar_w * progress)
            cv2.rectangle(frame, (10, y_base), (10 + bar_w, y_base + 12), (60, 60, 60), -1)
            bar_color = (0, 255, 180) if progress >= 1.0 else (0, 180, 255)
            if filled > 0:
                cv2.rectangle(frame, (10, y_base), (10 + filled, y_base + 12), bar_color, -1)
            cv2.putText(
                frame, f"{int(progress * 100)}%",
                (bar_w + 18, y_base + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1,
            )

        # Info de debug
        if self._cfg.SHOW_DEBUG_INFO and state.debug_info:
            y_d = y_base + 30
            for hand_label, info in state.debug_info.items():
                txt = (
                    f"{hand_label}: y={info['wrist_y']:.2f} "
                    f"ext={info['extended']} curl={info['curled']}"
                )
                cv2.putText(
                    frame, txt,
                    (10, y_d),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1,
                )
                y_d += 18

        # Banner grande cuando la pose está activa
        if state.cat_pose_active:
            banner = "** MemCAT **"
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
            bx = (w - bw) // 2
            by = 60
            # Sombra
            cv2.putText(frame, banner, (bx + 2, by + 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 80, 40), 3)
            # Texto
            cv2.putText(frame, banner, (bx, by),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 180), 3)

        return frame
