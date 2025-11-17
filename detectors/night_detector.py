# detectors/night_detector.py
import cv2
import mediapipe as mp
from detectors.fall_logic import FallLogic

class NightDetector:
    def __init__(self):
        self.fall_logic = FallLogic()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3
        )

    def apply_thermal(self, frame):
        """
        Convierte el frame en una imagen tipo “térmica” para mejorar contraste de noche.
        Puedes ajustarlo a tu gusto.
        """
        # Pasamos a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Igualamos histograma para realzar zonas oscuras
        gray = cv2.equalizeHist(gray)

        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return thermal

    def process(self, frame):
        """
        Devuelve:
        - fall: bool
        - out_frame: frame térmico con anotaciones
        - angle: ángulo del torso
        - velocity: velocidad de bajada de la cabeza
        """
        # Se aplica el efecto térmico
        thermal_img = self.apply_thermal(frame)
        height, width = thermal_img.shape[:2]

        rgb = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        fall = False
        angle = None
        velocity = None

        if results and results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                thermal_img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=3, circle_radius=3),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(255, 255, 255), thickness=2)
            )

            fall, _, _ = self.fall_logic.detect_fall(results, height)

        # Línea divisoria usando el mismo ratio de FallLogic
        line_y = int(height * self.fall_logic.line_ratio)
        cv2.line(thermal_img, (0, line_y), (width, line_y), (0, 0, 255), 2)

        return fall, thermal_img, angle, velocity