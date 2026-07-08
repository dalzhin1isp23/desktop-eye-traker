import pyautogui
import time
import cv2
import mediapipe as mp
import numpy as np

class EyeCommand:
    def __init__(self, click_delay=0.8):  # Увеличил задержку
        self.click_delay = click_delay
        self.last_click_time = 0
        self.last_blink_time = 0
        self.left_eye_open = True
        self.right_eye_open = True
        
        # Пороги для определения закрытых глаз
        self.EYE_CLOSED_THRESHOLD = 0.20  # Снизил порог
        self.BLINK_COOLDOWN = 0.5  # Увеличил кулдаун

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Индексы для EAR (Eye Aspect Ratio)
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        
        # История для сглаживания
        self.ear_history_left = []
        self.ear_history_right = []

    def _euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _eye_aspect_ratio(self, eye_points):
        """Вычисляет отношение сторон глаза (EAR)"""
        # Вертикальные расстояния
        A = self._euclidean_distance(eye_points[1], eye_points[5])
        B = self._euclidean_distance(eye_points[2], eye_points[4])
        
        # Горизонтальное расстояние
        C = self._euclidean_distance(eye_points[0], eye_points[3])
        
        # EAR формула
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear

    def _smooth_ear(self, ear, history_list, window=5):
        """Сглаживание EAR для уменьшения шума"""
        history_list.append(ear)
        if len(history_list) > window:
            history_list.pop(0)
        
        if len(history_list) >= 3:
            # Используем медианный фильтр
            return np.median(history_list[-3:])
        return ear

    def process(self, frame):
        """Обрабатывает кадр для определения команд"""
        if frame is None or frame.size == 0:
            return

        current_time = time.time()
        
        # Проверяем кулдаун после моргания двумя глазами
        if current_time - self.last_blink_time < self.BLINK_COOLDOWN:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Получаем координаты точек глаз
        left_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in self.LEFT_EYE_IDX]
        right_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in self.RIGHT_EYE_IDX]

        # Вычисляем EAR для каждого глаза
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        
        # Сглаживаем значения
        left_ear_smooth = self._smooth_ear(left_ear, self.ear_history_left)
        right_ear_smooth = self._smooth_ear(right_ear, self.ear_history_right)

        # Определяем состояние глаз
        left_closed = left_ear_smooth < self.EYE_CLOSED_THRESHOLD
        right_closed = right_ear_smooth < self.EYE_CLOSED_THRESHOLD

        # Обрабатываем моргание двумя глазами (игнорируем)
        if left_closed and right_closed:
            self.last_blink_time = current_time
            print(f"[Команда] Моргание двумя глазами (игнорируется)")
            return

        # Обрабатываем команды
        if current_time - self.last_click_time > self.click_delay:
            # Левый клик: левый глаз закрыт, правый открыт
            if left_closed and not right_closed:
                print(f"[Команда] Левый клик (LEAR={left_ear_smooth:.2f}, REAR={right_ear_smooth:.2f})")
                pyautogui.click(button='left')
                self.last_click_time = current_time
            
            # Правый клик: правый глаз закрыт, левый открыт
            elif right_closed and not left_closed:
                print(f"[Команда] Правый клик (LEAR={left_ear_smooth:.2f}, REAR={right_ear_smooth:.2f})")
                pyautogui.click(button='right')
                self.last_click_time = current_time

        # Сохраняем текущее состояние
        self.left_eye_open = not left_closed
        self.right_eye_open = not right_closed