# calibrator.py
import json
import numpy as np
import os

class Calibrator:
    def __init__(self):
        self.is_calibrated = False
        self.calibration_data = []
        self.model_x = None
        self.model_y = None
        self.screen_size = None
        self.load_calibration()

    def load_calibration(self, filepath="calibration.json"):
        try:
            if not os.path.exists(filepath):
                print(f"[Калибровка] Файл {filepath} не найден")
                self.is_calibrated = False
                return False

            with open(filepath, "r") as f:
                data = json.load(f)

            print(f"[Калибровка] Загружаю данные из {filepath}")

            # Поддержка нового формата с явными параметрами модели
            if "model_x" in data and "model_y" in data:
                self.model_x = data["model_x"]
                self.model_y = data["model_y"]
                self.screen_size = (
                    data.get("screen_width", 1920),
                    data.get("screen_height", 1080)
                )
                self.is_calibrated = True
                print("[Калибровка] Загружена модель с ручным масштабированием Y")
                print(f"[Калибровка] Параметры X: a={self.model_x[0]:.2f}, b={self.model_x[1]:.1f}")
                print(f"[Калибровка] Параметры Y: a={self.model_y[0]:.2f}, b={self.model_y[1]:.1f}")
                return True

            # Старый формат — попытка пересоздать с растяжением Y
            elif "points" in data:
                points = data["points"]
                if len(points) < 3:
                    print("[Калибровка] Недостаточно точек")
                    self.is_calibrated = False
                    return False

                # Извлекаем данные
                gaze_x = np.array([float(p["gaze_x"]) for p in points])
                gaze_y = np.array([float(p["gaze_y"]) for p in points])
                screen_x = np.array([float(p["screen_x"]) for p in points])
                screen_y = np.array([float(p["screen_y"]) for p in points])

                # === X: линейная регрессия (как раньше) ===
                A_x = np.vstack([gaze_x, np.ones(len(gaze_x))]).T
                self.model_x = np.linalg.lstsq(A_x, screen_x, rcond=None)[0]

                # === Y: ручное растяжение на диапазон экрана ===
                gaze_y_min, gaze_y_max = gaze_y.min(), gaze_y.max()
                screen_y_min, screen_y_max = screen_y.min(), screen_y.max()

                if gaze_y_max - gaze_y_min < 1e-8:
                    print("[Калибровка] Предупреждение: gaze_y не меняется — невозможно калибровать Y")
                    self.model_y = [0.0, float(screen_y.mean())]
                else:
                    # a * gaze_y + b = screen_y
                    # При gaze_y = gaze_y_min → screen_y = screen_y_max (взгляд вверх = верх экрана)
                    # При gaze_y = gaze_y_max → screen_y = screen_y_min (взгляд вниз = низ экрана)
                    a_y = (screen_y_min - screen_y_max) / (gaze_y_max - gaze_y_min)
                    b_y = screen_y_max - a_y * gaze_y_min
                    self.model_y = [a_y, b_y]

                self.screen_size = (
                    data.get("screen_width", 1920),
                    data.get("screen_height", 1080)
                )
                self.is_calibrated = True

                print("[Калибровка] Пересоздана модель с растяжением Y")
                print(f"[Калибровка] Параметры X: a={self.model_x[0]:.2f}, b={self.model_x[1]:.1f}")
                print(f"[Калибровка] Параметры Y: a={self.model_y[0]:.2f}, b={self.model_y[1]:.1f}")

                # Сохраняем обновлённую модель
                self._save_model(data["points"], self.screen_size)
                return True

            else:
                print("[Калибровка] Неизвестный формат файла")
                self.is_calibrated = False
                return False

        except Exception as e:
            print(f"[Калибровка] Ошибка загрузки: {e}")
            self.is_calibrated = False
            return False

    def _save_model(self, points, screen_size):
        """Сохраняет модель в новый формат."""
        try:
            calib_dict = {
                "points": points,
                "screen_width": int(screen_size[0]),
                "screen_height": int(screen_size[1]),
                "model_x": [float(self.model_x[0]), float(self.model_x[1])],
                "model_y": [float(self.model_y[0]), float(self.model_y[1])]
            }
            with open("calibration.json", "w") as f:
                json.dump(calib_dict, f, indent=2)
            print("[Калибровка] Модель сохранена в новом формате")
        except Exception as e:
            print(f"[Калибровка] Ошибка сохранения модели: {e}")

    def calibrate_from_points(self, calibration_data, screen_size):
        """calibration_data = [(gaze_offset, screen_point), ...]"""
        if len(calibration_data) < 3:
            print("[Калибровка] Недостаточно точек для калибровки (минимум 3)")
            return False

        print(f"[Калибровка] Обрабатываю {len(calibration_data)} точек")

        # Извлекаем массивы
        gaze_x = np.array([float(gaze[0]) for gaze, _ in calibration_data])
        gaze_y = np.array([float(gaze[1]) for gaze, _ in calibration_data])
        screen_x = np.array([float(screen[0]) for _, screen in calibration_data])
        screen_y = np.array([float(screen[1]) for _, screen in calibration_data])

        # === Калибровка X: линейная регрессия ===
        A_x = np.vstack([gaze_x, np.ones(len(gaze_x))]).T
        self.model_x = np.linalg.lstsq(A_x, screen_x, rcond=None)[0]

        # === Калибровка Y: растягиваем на диапазон экрана ===
        gaze_y_min, gaze_y_max = gaze_y.min(), gaze_y.max()
        screen_y_min, screen_y_max = screen_y.min(), screen_y.max()

        if gaze_y_max - gaze_y_min < 1e-8:
            print("[Калибровка] Внимание: gaze_y почти не меняется! Y будет фиксированным.")
            self.model_y = [0.0, float(screen_y.mean())]
        else:
            # Инвертируем: взгляд вверх (меньше gaze_y) → верх экрана (меньше screen_y)
            a_y = (screen_y_min - screen_y_max) / (gaze_y_max - gaze_y_min)
            b_y = screen_y_max - a_y * gaze_y_min
            self.model_y = [a_y, b_y]

        self.screen_size = screen_size
        self.is_calibrated = True

        # Сохраняем сырые данные
        raw_data = []
        for (gaze, screen) in calibration_data:
            raw_data.append({
                "gaze_x": float(gaze[0]),
                "gaze_y": float(gaze[1]),
                "screen_x": int(screen[0]),
                "screen_y": int(screen[1])
            })

        # Сохраняем всё вместе с моделью
        self._save_model(raw_data, screen_size)

        print("[Калибровка] Успешно завершена с растяжением Y!")
        print(f"[Калибровка] Параметры X: a={self.model_x[0]:.2f}, b={self.model_x[1]:.1f}")
        print(f"[Калибровка] Параметры Y: a={self.model_y[0]:.2f}, b={self.model_y[1]:.1f}")
        return True

    def map_gaze_to_screen(self, offset, screen_size):
        if not self.is_calibrated or offset is None or self.model_x is None or self.model_y is None:
            return None

        gx, gy = offset
        sx = int(self.model_x[0] * gx + self.model_x[1])
        sy = int(self.model_y[0] * gy + self.model_y[1])

        # Ограничение границ
        margin = 10
        sx = np.clip(sx, margin, screen_size[0] - margin - 1)
        sy = np.clip(sy, margin, screen_size[1] - margin - 1)

        return (sx, sy)