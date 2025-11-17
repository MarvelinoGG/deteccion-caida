import cv2
import mediapipe as mp
from utils.face_filter import FaceFilter
from detectors.fall_logic import FallLogic


class DayDetector:
    def __init__(self):
        self.face_filter = FaceFilter()
        self.fall_logic = FallLogic()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.consecutive_no_face_frames = 0
        self.max_no_face_frames = 5  # Tolerancia antes de descartar

    def process(self, frame):
        """
        Devuelve:
        - fall: bool
        - out_frame: frame con anotaciones
        - angle: ángulo del torso (para debug)
        - velocity: velocidad de bajada de la cabeza (para debug)
        """
        out = frame.copy()
        height, width = out.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detección de cara (para evitar analizar frames totalmente vacíos)
        faces = self.face_filter.get_face_regions(gray)
        if not faces:
            self.consecutive_no_face_frames += 1
            if self.consecutive_no_face_frames > self.max_no_face_frames:
                # Reseteamos estado de caída si llevamos muchos frames sin cara
                self.fall_logic.reset()
            # Dibujamos la línea igualmente
            line_y = int(height * self.fall_logic.line_ratio)
            cv2.line(out, (0, line_y), (width, line_y), (0, 0, 255), 2)
            return False, out, None, None
        else:
            self.consecutive_no_face_frames = 0

        # Para debug: dibujar rectángulo de la cara principal
        x, y, w, h = faces[0]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # MediaPipe Pose trabaja en RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        fall = False
        angle = None
        velocity = None

        if results and results.pose_landmarks:
            # Opcional: filtrar poses que no parecen humanas
            if self.is_human_pose(results):
                fall, _, _ = self.fall_logic.detect_fall(results, height)

            # Dibujar pose
            mp.solutions.drawing_utils.draw_landmarks(
                out,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(255, 255, 255), thickness=2)
            )

        # Dibujar línea divisoria (mitad de la pantalla, o donde marque FallLogic.line_ratio)
        line_y = int(height * self.fall_logic.line_ratio)
        cv2.line(out, (0, line_y), (width, line_y), (0, 0, 255), 2)

        return fall, out, angle, velocity

    def is_human_pose(self, results):
        landmarks = results.pose_landmarks.landmark
        
        # Verificar estructura básica humana
        try:
            # Hombros arriba de caderas (en coordenadas Y)
            shoulders_above_hips = (landmarks[11].y + landmarks[12].y) < (landmarks[23].y + landmarks[24].y)
            
            # Caderas arriba de rodillas
            hips_above_knees = (landmarks[23].y + landmarks[24].y) < (landmarks[25].y + landmarks[26].y)
            
            return shoulders_above_hips and hips_above_knees
            
        except IndexError:
            return False