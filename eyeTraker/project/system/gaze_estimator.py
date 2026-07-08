# gaze_estimator.py
import cv2
import mediapipe as mp
import numpy as np

class GazeEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Ключевые точки (MediaPipe Face Mesh indices)
        # Радужки — центры
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        # Контуры глаз для определения границ
        self.LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        
        # Точки для верхнего и нижнего века (для вертикали)
        # Левый глаз
        self.LEFT_EYE_TOP = [159, 158, 160]    # верхние точки
        self.LEFT_EYE_BOTTOM = [145, 153, 144] # нижние точки
        # Правый глаз
        self.RIGHT_EYE_TOP = [386, 385, 387]
        self.RIGHT_EYE_BOTTOM = [374, 380, 373]

        # Пороги моргания (EAR — Eye Aspect Ratio)
        self.EAR_THRESHOLD = 0.25  # ниже — считаем, что глаз закрыт или искажён

    def _get_point(self, landmarks, idx):
        """Возвращает (x, y) нормализованную точку по индексу."""
        lm = landmarks[idx]
        return np.array([lm.x, lm.y], dtype=np.float32)

    def _eye_center(self, landmarks, indices):
        """Вычисляет центр по группе точек."""
        points = np.array([self._get_point(landmarks, i) for i in indices])
        return np.mean(points, axis=0)

    def _eye_aspect_ratio(self, eye_contour_points):
        """Вычисляет EAR (Eye Aspect Ratio) для фильтрации морганий."""
        # Расстояния между вертикальными маркерами
        A = np.linalg.norm(eye_contour_points[1] - eye_contour_points[5])
        B = np.linalg.norm(eye_contour_points[2] - eye_contour_points[4])
        # Расстояние между горизонтальными маркерами
        C = np.linalg.norm(eye_contour_points[0] - eye_contour_points[3])
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear

    def _normalize_gaze_in_eye(self, iris, top_points, bottom_points, left_point, right_point):
        """
        Нормализует положение радужки внутри прямоугольника глаза.
        Возвращает (norm_x, norm_y) в диапазоне [-1, 1], где (0,0) — центр.
        """
        # Горизонтальные границы
        min_x = min(left_point[0], right_point[0])
        max_x = max(left_point[0], right_point[0])
        width = max_x - min_x + 1e-6

        # Вертикальные границы
        top_y = min(p[1] for p in top_points)
        bottom_y = max(p[1] for p in bottom_points)
        height = bottom_y - top_y + 1e-6

        # Нормализуем положение радужки
        norm_x = (iris[0] - min_x) / width
        norm_y = (iris[1] - top_y) / height

        # Преобразуем в [-1, 1], где 0 — центр
        gaze_x = 2.0 * (norm_x - 0.5)
        gaze_y = 2.0 * (norm_y - 0.5)

        return np.clip(gaze_x, -1.0, 1.0), np.clip(gaze_y, -1.0, 1.0)

    def estimate_gaze(self, frame):
        """
        Оценивает направление взгляда.
        Возвращает: (gaze_x, gaze_y) в диапазоне [-1, 1],
        где (0,0) — центр, (-1,-1) — вверх-влево, (1,1) — вниз-вправо.
        Возвращает None, если лицо не найдено или глаза искажены.
        """
        if frame is None or frame.size == 0:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        try:
            # === Получаем контуры глаз ===
            left_contour = np.array([self._get_point(landmarks, i) for i in self.LEFT_EYE_CONTOUR])
            right_contour = np.array([self._get_point(landmarks, i) for i in self.RIGHT_EYE_CONTOUR])

            # === Фильтрация: проверяем, не закрыты ли глаза ===
            left_ear = self._eye_aspect_ratio(left_contour)
            right_ear = self._eye_aspect_ratio(right_contour)

            if left_ear < self.EAR_THRESHOLD or right_ear < self.EAR_THRESHOLD:
                # Глаза слишком закрыты/искажены — пропускаем кадр
                return None

            # === Центры радужек ===
            left_iris = self._eye_center(landmarks, self.LEFT_IRIS)
            right_iris = self._eye_center(landmarks, self.RIGHT_IRIS)

            # === Края глаз ===
            left_left = self._get_point(landmarks, 33)
            left_right = self._get_point(landmarks, 133)
            right_left = self._get_point(landmarks, 362)
            right_right = self._get_point(landmarks, 263)

            # === Верхние и нижние точки ===
            left_top_pts = [self._get_point(landmarks, i) for i in self.LEFT_EYE_TOP]
            left_bottom_pts = [self._get_point(landmarks, i) for i in self.LEFT_EYE_BOTTOM]
            right_top_pts = [self._get_point(landmarks, i) for i in self.RIGHT_EYE_TOP]
            right_bottom_pts = [self._get_point(landmarks, i) for i in self.RIGHT_EYE_BOTTOM]

            # === Нормализуем взгляд для каждого глаза ===
            gaze_left_x, gaze_left_y = self._normalize_gaze_in_eye(
                left_iris, left_top_pts, left_bottom_pts, left_left, left_right
            )
            gaze_right_x, gaze_right_y = self._normalize_gaze_in_eye(
                right_iris, right_top_pts, right_bottom_pts, right_left, right_right
            )

            # === Усредняем два глаза ===
            gaze_x = (gaze_left_x + gaze_right_x) / 2.0
            gaze_y = (gaze_left_y + gaze_right_y) / 2.0

            return (float(gaze_x), float(gaze_y))

        except Exception as e:
            # print(f"[GazeEstimator] Ошибка: {e}")  # можно включить для отладки
            return None