# camera/phonecam.py
import cv2
import requests
import numpy as np

class PhoneCamera:
    def __init__(self, url="http://7.72.213.173:8080/shot.jpg"):
        """
        Использует /shot.jpg — одиночные кадры от IP Webcam.
        Это надёжнее, чем /video или /mjpeg.
        """
        self.base_url = url  # например: http://192.168.1.100:8080/shot.jpg
        self.session = requests.Session()

    def connect(self) -> bool:
        try:
            resp = self.session.get(self.base_url, timeout=3)
            return resp.status_code == 200
        except Exception as e:
            print(f"[PhoneCamera] Ошибка подключения: {e}")
            return False

    def read_frame(self):
        try:
            resp = self.session.get(self.base_url, timeout=2)
            if resp.status_code == 200:
                arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return frame
        except Exception as e:
            print(f"[PhoneCamera] Ошибка чтения кадра: {e}")
            return None

    def disconnect(self):
        self.session.close()