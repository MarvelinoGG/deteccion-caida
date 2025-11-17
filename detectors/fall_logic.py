# detectors/fall_logic.py
import numpy as np
import time

class FallLogic:
    def __init__(
        self,
        line_ratio: float = 0.6,       # línea a 1/3 de la altura
        frames_below_line_needed: int = 10,  # frames seguidos por debajo para confirmar caída
        max_gap_frames: int = 60            # tras cuántos frames sin pose reseteamos todo
    ):
        self.line_ratio = line_ratio
        self.frames_below_line_needed = frames_below_line_needed
        self.max_gap_frames = max_gap_frames

        # Estado interno
        self.last_position = None          # "above" o "below" la última vez que vimos la cabeza
        self.frames_below_line = 0         # cuántos frames llevamos por debajo desde el cruce
        self.missed_frames = 0             # frames seguidos sin pose
        self.fall_confirmed = False        # si ya hemos confirmado la caída

    def reset(self):
        self.last_position = None
        self.frames_below_line = 0
        self.missed_frames = 0
        self.fall_confirmed = False

    def detect_fall(self, results, frame_height: int):
        """
        Lógica muy simple basada SOLO en la cabeza (nariz):
        - Si la cabeza pasa de estar por ENCIMA de la línea a por DEBAJO, decimos que ha "cruzado".
        - Si, después del cruce, permanece N frames seguidos por debajo, marcamos caída.

        También:
        - Si la cara se pierde unos pocos frames, mantenemos el estado.
        - Si se pierde demasiados frames seguidos (> max_gap_frames), reseteamos todo.

        Devuelve: (fall: bool, angle: None, velocity: None)
        """

        # 1) Si no hay pose → contamos frame perdido y devolvemos estado actual
        if not results or not results.pose_landmarks:
            self.missed_frames += 1
            if self.missed_frames > self.max_gap_frames:
                # demasiados frames sin ver a la persona → reseteamos
                self.reset()
            # devolvemos si estaba ya confirmada una caída o no
            return self.fall_confirmed, None, None

        # Si hay pose, reseteamos contador de frames perdidos
        self.missed_frames = 0

        landmarks = results.pose_landmarks.landmark

        # MediaPipe Pose: 0 = nariz
        head_y = landmarks[0].y  # normalizado [0,1], 0 arriba, 1 abajo

        # 2) Posición actual respecto a la línea
        current_position = "below" if head_y >= self.line_ratio else "above"

        # 3) Detectar cruce hacia abajo (de above -> below)
        crossed_down = False
        if self.last_position is not None:
            if self.last_position == "above" and current_position == "below":
                crossed_down = True

        # 4) Actualizamos conteo bajo la línea según el cruce/posición actual
        if crossed_down:
            # justo ahora hemos cruzado hacia abajo
            self.frames_below_line = 1
            self.fall_confirmed = False
        else:
            if current_position == "below" and self.frames_below_line > 0:
                # seguimos por debajo desde un cruce anterior
                self.frames_below_line += 1
            elif current_position == "above":
                # ha subido otra vez antes de llegar a N frames → reseteamos conteo
                self.frames_below_line = 0
                self.fall_confirmed = False
            # Si current_position == "below" pero frames_below_line == 0,
            # significa que nunca detectamos cruce (por ejemplo empezó ya debajo),
            # NO lo contamos como caída por simplicidad.

        # 5) Confirmamos caída si se ha mantenido suficientes frames abajo
        if self.frames_below_line >= self.frames_below_line_needed:
            self.fall_confirmed = True

        # 6) Guardamos posición actual para el siguiente frame
        self.last_position = current_position

        return self.fall_confirmed, None, None


