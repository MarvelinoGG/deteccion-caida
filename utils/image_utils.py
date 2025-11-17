# utils/image_utils.py
import cv2

def enhance_night(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    equalized = cv2.equalizeHist(denoised)
    return equalized