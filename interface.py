# gui.py
import sys
import cv2
import mediapipe as mp
from ultralytics import YOLO
from alarm.alarm import play_alarm
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QListWidget
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from detectors.day_detector import DayDetector
from detectors.night_detector import NightDetector
from utils.camera_manager import CameraManager
from alarm.telegram_alarm import send_fall_alert_with_location

class FallDetectionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Detección de Caídas")
        self.setGeometry(100, 100, 1000, 600)

        # --- Widgets principales ---
        self.video_label = QLabel(self)
        self.status_label = QLabel("Estado: Inactivo")
        self.angle_label = QLabel("Ángulo: -")
        self.vel_label = QLabel("Velocidad: -")
        self.alert_label = QLabel("")
        self.alert_label.setStyleSheet("color: red; font-size: 18px; font-weight: bold;")

        self.history_list = QListWidget()

        self.start_button = QPushButton("Iniciar detección")
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton("Detener detección")
        self.stop_button.clicked.connect(self.stop_detection)

        # --- Layout ---
        side_layout = QVBoxLayout()
        side_layout.addWidget(self.status_label)
        side_layout.addWidget(self.angle_label)
        side_layout.addWidget(self.vel_label)
        side_layout.addWidget(self.alert_label)
        side_layout.addWidget(QLabel("Historial de caídas:"))
        side_layout.addWidget(self.history_list)
        side_layout.addWidget(self.start_button)
        side_layout.addWidget(self.stop_button)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label, 3)
        main_layout.addLayout(side_layout, 1)

        self.setLayout(main_layout)

        # --- Lógica detección ---
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cam_manager = CameraManager()
        self.day = DayDetector()
        self.night = NightDetector()

    def start_detection(self):
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(1,  cv2.CAP_DSHOW)
        self.timer.start(30)
        self.status_label.setText("Estado: Detectando...")

    def stop_detection(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.status_label.setText("Estado: Inactivo")

    def detect_persons(self, frame):
        # Cargar el modelo YOLOv8 preentrenado
        model = YOLO("yolov8n.pt")  # Puedes usar otros modelos como yolov5m.pt o yolov5l.pt

        # Realizar la detección
        results = model.predict(source=frame, conf=0.5, iou=0.45, classes=[0], device="cpu", show=False)

        persons = []
        h, w, _ = frame.shape

        # Procesar las detecciones
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf.cpu().numpy()

                # Filtrar detecciones con alta confianza
                if confidence > 0.5:
                    persons.append((x_min, y_min, x_max - x_min, y_max - y_min))  # ROI de la persona

        return persons

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        persons = self.detect_persons(frame)  # Detectar personas con YOLO

        # Inicializar MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        for idx, person in enumerate(persons):
            x, y, w, h = person  # Coordenadas del ROI de la persona

            # Validar que el ROI esté dentro de los límites del frame
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue  # Ignorar detecciones fuera de los límites

            # Recortar el ROI completo (bounding box de la persona)
            roi = frame[y:y+h, x:x+w]

            # Convertir el ROI a RGB (MediaPipe requiere RGB)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Procesar el ROI con MediaPipe Pose
            results = pose.process(roi_rgb)

            if results.pose_landmarks:
                # Usar los métodos existentes para procesar los puntos clave
                if self.cam_manager.is_night(frame):
                    fall, img, angle, vel = self.night.process(frame)
                    out_frame = img
                    mode = "Modo noche (MediaPipe + Haar)"
                else:
                    fall, img, angle, vel = self.day.process(frame)
                    out_frame = img
                    mode = "Modo dia (MediaPipe + Haar)"

                rgb_image = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
                # else:
                #     fall, img, angle, vel = self.day.process(frame)
                #     mode = "Modo dia (MediaPipe + Haar)"

                # Mostrar datos para cada persona
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Dibujar el bounding box completo
                cv2.putText(frame, f"Persona {idx + 1}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, mode, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if fall:
                    self.alert_label.setText("🚨 CAÍDA DETECTADA 🚨")
                    print("NIIII NOOOO NIIII NOOOO")  # ALERTA
                    send_fall_alert_with_location()
                    play_alarm()
                    self.history_list.addItem(f"Caída confirmada - Persona {idx + 1} - {time.strftime('%H:%M:%S')}")
                else:
                    self.alert_label.setText("")

        # Convertir frame a formato Qt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = FallDetectionGUI()
    gui.show()
    sys.exit(app.exec_())