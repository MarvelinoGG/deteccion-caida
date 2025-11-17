import cv2
import os
import numpy as np

class FaceFilter:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(base_dir, "..", "cascades", "haarcascade_frontalface_default.xml")
        cascade_path = os.path.normpath(cascade_path)

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    # Verificar que el archivo existe
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"No se encuentra el cascade: {cascade_path}")
        
        #self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Verificar que el cascade cargó correctamente
        if self.face_cascade.empty():
            raise ValueError("No se pudo cargar el clasificador Haar")
        
        self.detection_cache = []
        self.cache_size = 3

    def detect_face(self, gray_frame):
        """Detección facial con filtrado temporal"""
        try:
            # Preprocesamiento para mejorar detección
            enhanced_frame = self._enhance_image(gray_frame)
            
            # Detección múltiple con diferentes parámetros
            faces = self._multi_scale_detection(enhanced_frame)
            
            # Filtrar detecciones falsas
            valid_faces = self._filter_false_positives(faces, enhanced_frame)
            
            # Usar cache para estabilizar detecciones
            self.detection_cache.append(len(valid_faces) > 0)
            if len(self.detection_cache) > self.cache_size:
                self.detection_cache.pop(0)
            
            # Consenso: mayoría de frames recientes deben tener cara
            face_detected = sum(self.detection_cache) >= (self.cache_size // 2)
            
            return face_detected
            
        except Exception as e:
            print(f"Error en detección facial: {e}")
            return False

    def _enhance_image(self, gray_frame):
        """Mejora la imagen para mejor detección"""
        # Ecualizar histograma
        equalized = cv2.equalizeHist(gray_frame)
        
        # Suavizado para reducir ruido
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        return blurred

    def _multi_scale_detection(self, gray_frame):
        """Detección con múltiples escalas y parámetros"""
        all_faces = []
        
        # Diferentes configuraciones para varios escenarios
        configs = [
            {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (40, 40)},
            {"scaleFactor": 1.2, "minNeighbors": 6, "minSize": (50, 50)},
            {"scaleFactor": 1.05, "minNeighbors": 4, "minSize": (30, 30)}
        ]
        
        for config in configs:
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=config["scaleFactor"],
                minNeighbors=config["minNeighbors"],
                minSize=config["minSize"],
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            all_faces.extend(faces)
        
        return all_faces

    def _filter_false_positives(self, faces, gray_frame):
        """Filtra detecciones falsas basado en características faciales"""
        valid_faces = []
        
        for (x, y, w, h) in faces:
            # Verificar relación de aspecto facial típica
            aspect_ratio = w / h
            if not (0.7 < aspect_ratio < 1.5):
                continue
                
            # Verificar tamaño mínimo/máximo razonable
            frame_area = gray_frame.shape[0] * gray_frame.shape[1]
            face_area = w * h
            face_ratio = face_area / frame_area
            
            if not (0.01 < face_ratio < 0.3):  # Entre 1% y 30% del frame
                continue
            
            # Verificar región de interés (ROI) para características faciales
            roi = gray_frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            # Verificar contraste en la región facial
            if np.std(roi) < 20:  # Muy poca variación = probable falso positivo
                continue
                
            valid_faces.append((x, y, w, h))
        
        return valid_faces

    def get_face_regions(self, gray_frame):
        """Opcional: obtener regiones faciales para debug"""
        faces = self._multi_scale_detection(gray_frame)
        valid_faces = self._filter_false_positives(faces, gray_frame)
        return valid_faces