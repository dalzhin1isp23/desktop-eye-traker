import cv2

class Webcam:
    def __init__(self, cam_id=0):
        self.cam_id = cam_id
        self.cap = None
        self.is_connected = False

    def connect(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.cam_id)
            if not self.cap.isOpened():
                print(f"[Webcam] Не удалось открыть камеру {self.cam_id}")
                return False
            
            # Устанавливаем параметры
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_connected = True
            print(f"[Webcam] Камера {self.cam_id} подключена")
            return True
            
        except Exception as e:
            print(f"[Webcam] Ошибка подключения: {e}")
            return False

    def read_frame(self):
        if not self.is_connected or self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            print("[Webcam] Ошибка чтения кадра")
            return None
            
        # Переворачиваем кадр по горизонтали для зеркального отображения
        frame = cv2.flip(frame, 1)
        return frame

    def disconnect(self):
        if self.cap is not None:
            self.cap.release()
            self.is_connected = False
            print("[Webcam] Камера отключена")