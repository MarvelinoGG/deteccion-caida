import cv2
import numpy as np

class CameraManager:
    def __init__(self):
        self.night_frame_count = 0
        self.required_night_frames = 15 # Frames para cambiar a nocturno
        
    def is_night(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        contrast = gray.std()
        
        # Condiciones MÁS ESTRICTAS para noche
        is_dark = (brightness < 50 and contrast < 40)  
        self.required_night_frames = 5

        # Cambio más lento entre modos
        if is_dark:
            self.night_frame_count = min(self.night_frame_count + 1, 30)
        else:
            self.night_frame_count = max(self.night_frame_count - 1, 0)
            
        return self.night_frame_count >= self.required_night_frames