# main.py
import cv2
from detectors.day_detector import DayDetector
from detectors.night_detector import NightDetector
from utils.camera_manager import CameraManager
from alarm.alarm import play_alarm
from alarm.telegram_alarm import send_fall_alert_with_location


cam = CameraManager()
day = DayDetector()
night = NightDetector()

# cap = cv2.VideoCapture(0) # camara integrada del pc
cap = cv2.VideoCapture(1,  cv2.CAP_DSHOW) # camara usb externa

alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Decidir modo día/noche
    is_night = cam.is_night(frame)

    if is_night:
        fall, img, angle, vel = night.process(frame)
        mode_text = "NOCHE"
    else:
        fall, img, angle, vel = day.process(frame)
        mode_text = "DIA"

    # Info en pantalla (modo, ángulo y velocidad)
    cv2.putText(img, mode_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # Sistema de confirmación de caída
    if fall:
        cv2.putText(img, "CAIDA DETECTADA", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        if not alarm_playing:
            # play_alarm()
            print("-------------ALARMA ACTIVADA")
            for i in range(5):
                print("NIIII NOOOO NIIII NOOOO")  # Sonido de alerta simple
            send_fall_alert_with_location()
            alarm_playing = True
    else:
        alarm_playing = False  # o añade lógica para silenciar tras X tiempo

    cv2.imshow("Sistema de Detección de Caídas", img)
    if cv2.waitKey(10) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()