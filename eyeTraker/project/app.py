# main.py
import tkinter as tk
import pyautogui
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import os

pyautogui.FAILSAFE = False

# Исправленные импорты - используйте относительные импорты или правильные пути
# Вариант 1: Если файлы находятся в тех же папках
from camera.webcam import Webcam
from system.gaze_estimator import GazeEstimator
from system.calibrator import Calibrator
from system.command import EyeCommand

# Вариант 2: Если импорты не работают, попробуйте такую структуру:
"""
project/
├── main.py
├── camera/
│   └── webcam.py
└── system/
    ├── gaze_estimator.py
    ├── calibrator.py
    └── command.py
"""

# 9 точек калибровки (центр, края, углы)
CALIBRATION_POINTS = [
    (0.5, 0.5),   # центр
    (0.1, 0.5),   # левый край
    (0.9, 0.5),   # правый край
    (0.5, 0.1),   # верхний край
    (0.5, 0.9),   # нижний край
    (0.1, 0.1),   # левый верхний угол
    (0.9, 0.1),   # правый верхний угол
    (0.1, 0.9),   # левый нижний угол
    (0.9, 0.9)    # правый нижний угол
]

class CalibrationPointWindow:
    """Отдельное окно для отображения точки калибровки"""
    def __init__(self, root, x, y, point_num, total_points):
        self.window = tk.Toplevel(root)
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg='black')
        self.window.attributes('-topmost', True)
        
        self.canvas = tk.Canvas(self.window, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Большая красная точка
        self.canvas.create_oval(x-40, y-40, x+40, y+40, 
                              fill='red', outline='white', width=4)
        self.canvas.create_oval(x-20, y-20, x+20, y+20, 
                              fill='darkred', outline='')
        
        # Номер точки
        self.canvas.create_text(x, y-80, 
                              text=f"Точка {point_num}/{total_points}",
                              fill='white', font=("Arial", 20, "bold"))
        
        # Таймер
        self.time_left = 3
        self.timer_text = self.canvas.create_text(x, y+80,
                                                 text=f"Смотрите: {self.time_left} сек",
                                                 fill='white', font=("Arial", 16))
        
        # Запускаем таймер
        self.update_timer()
    
    def update_timer(self):
        if self.time_left > 0:
            self.canvas.itemconfig(self.timer_text, text=f"Смотрите: {self.time_left} сек")
            self.time_left -= 1
            self.window.after(1000, self.update_timer)
    
    def close(self):
        self.window.destroy()

class EyeTrackerApp:
    def __init__(self, root):
        self.root = root
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()

        self.gaze_estimator = GazeEstimator()
        self.calibrator = Calibrator()
        self.eye_command = EyeCommand(click_delay=0.5)
        
        # Загружаем калибровку если есть
        self.is_calibrated = self.calibrator.is_calibrated
        
        self.is_running = False
        self.is_calibrating = False
        self.tracking_enabled = self.is_calibrated
        
        self.current_camera = None
        self._last_gaze = None
        
        # Для калибровки
        self.calibration_points = CALIBRATION_POINTS
        self.current_point_index = 0
        self.current_samples = []
        self.calibration_data = []
        self.calibration_window = None
        
        # История для сглаживания
        self.gaze_history = []
        
        self.setup_ui()
    
    def setup_ui(self):
        root = self.root
        root.title("Eye Tracker")
        root.attributes('-fullscreen', True)
        root.configure(bg='black')
        root.bind('<Escape>', lambda e: root.destroy())
        root.bind('<space>', lambda e: self.toggle_tracking())
        
        self.video_label = tk.Label(root, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(
            root, text="Статус: Ожидание калибровки",
            bg='black', fg='yellow', font=("Arial", 16, "bold")
        )
        self.status_label.place(x=20, y=20)
        
        self.calibration_label = tk.Label(
            root, text="",
            bg='black', fg='white', font=("Arial", 14)
        )
        self.calibration_label.place(relx=0.5, y=60, anchor='center')
        
        btn_frame = tk.Frame(root, bg='black')
        btn_frame.place(relx=0.5, rely=0.95, anchor='center')
        
        self.calibrate_btn = tk.Button(
            btn_frame, text="🎯 Калибровка", 
            command=self.start_calibration,
            font=("Arial", 14), bg='#D32F2F', fg='white',
            activebackground='#F44336', activeforeground='white'
        )
        self.calibrate_btn.pack(side=tk.LEFT, padx=10)
        
        self.tracking_btn = tk.Button(
            btn_frame, text="🖱️ Включить управление", 
            command=self.toggle_tracking,
            font=("Arial", 14), bg='#6A1B9A', fg='white',
            activebackground='#9C27B0', activeforeground='white'
        )
        self.tracking_btn.pack(side=tk.LEFT, padx=10)
        
        self.exit_btn = tk.Button(
            btn_frame, text="❌ Выход", 
            command=self.cleanup_and_exit,
            font=("Arial", 14), bg='#424242', fg='white',
            activebackground='#616161', activeforeground='white'
        )
        self.exit_btn.pack(side=tk.LEFT, padx=10)
        
        if not self.is_calibrated:
            self.tracking_btn.config(state='disabled')
        
        self.update_status_label()
    
    def update_status_label(self):
        if not self.is_calibrated:
            text = "Статус: Ожидание калибровки"
            color = 'yellow'
        elif not self.tracking_enabled:
            text = "Статус: Калибровка завершена. Управление выключено"
            color = 'cyan'
        else:
            text = "Статус: Управление активно"
            color = 'green'
        
        self.status_label.config(text=text, fg=color)
    
    def toggle_tracking(self):
        if self.is_calibrated:
            self.tracking_enabled = not self.tracking_enabled
            status = "включено" if self.tracking_enabled else "выключено"
            print(f"[Управление] {status}")
            self.update_status_label()
            
            if self.tracking_enabled:
                self.tracking_btn.config(text="🖱️ Выключить управление")
            else:
                self.tracking_btn.config(text="🖱️ Включить управление")
        else:
            print("[Управление] Сначала выполните калибровку!")
    
    def start_calibration(self):
        if not self.is_running:
            print("[Калибровка] Включите камеру сначала")
            return
        if self.is_calibrating:
            print("[Калибровка] Уже запущена")
            return
        
        # Подключаем камеру если еще не подключена
        if self.current_camera is None:
            self.current_camera = Webcam(cam_id=0)
            if not self.current_camera.connect():
                print("[Калибровка] Не удалось подключить камеру")
                return
            self.is_running = True

        self.is_calibrating = True
        self.tracking_enabled = False
        self.current_point_index = 0
        self.calibration_data = []
        self.current_samples = []
        
        self.calibrate_btn.config(state='disabled')
        self.tracking_btn.config(state='disabled')
        
        print("[Калибровка] Начата калибровка")
        self.calibration_label.config(text="вставьте лицо в трафарет", fg='yellow')
        
        self.root.after(3000, self.next_calibration_step)
    
    def next_calibration_step(self):
        if not self.is_calibrating:
            return
        
        if self.current_point_index >= len(self.calibration_points):
            self.finish_calibration()
            return
        
        # Закрываем предыдущее окно если есть
        if self.calibration_window:
            self.calibration_window.close()
        
        # Получаем точку калибровки
        point_x, point_y = self.calibration_points[self.current_point_index]
        screen_x = int(point_x * self.screen_width)
        screen_y = int(point_y * self.screen_height)
        
        # Создаем окно с точкой калибровки
        self.calibration_window = CalibrationPointWindow(
            self.root, screen_x, screen_y,
            self.current_point_index + 1, len(self.calibration_points)
        )
        
        self.current_samples = []
        self.calibration_label.config(
            text=f"Смотрите на точку {self.current_point_index + 1}/{len(self.calibration_points)}", 
            fg='red'
        )
        
        # Запускаем сбор данных через 1 секунду (после отображения точки)
        self.root.after(1000, self.collect_calibration_data, screen_x, screen_y)
    
    def collect_calibration_data(self, target_x, target_y):
        """Собирает данные калибровки для текущей точки"""
        if not self.is_calibrating:
            return
        
        start_time = time.time()
        
        # Собираем данные в течение 3 секунд
        while time.time() - start_time < 3.0:
            frame = self.current_camera.read_frame()
            if frame is not None:
                gaze_offset = self.gaze_estimator.estimate_gaze(frame)
                if gaze_offset is not None:
                    self.current_samples.append(gaze_offset)
            
            # Небольшая задержка для стабильности
            time.sleep(0.033)  # ~30 FPS
        
        # Обрабатываем собранные данные
        if self.current_samples:
            # Усредняем значения, исключая выбросы
            samples_array = np.array(self.current_samples)
            
            # Удаляем выбросы (больше 2 стандартных отклонений)
            x_mean, x_std = np.mean(samples_array[:, 0]), np.std(samples_array[:, 0])
            y_mean, y_std = np.mean(samples_array[:, 1]), np.std(samples_array[:, 1])
            
            mask = (np.abs(samples_array[:, 0] - x_mean) < 2*x_std) & \
                   (np.abs(samples_array[:, 1] - y_mean) < 2*y_std)
            
            filtered_samples = samples_array[mask]
            
            if len(filtered_samples) > 0:
                avg_gaze = np.mean(filtered_samples, axis=0)
                self.calibration_data.append((avg_gaze, (target_x, target_y)))
                print(f"[Калибровка] Точка {self.current_point_index + 1} сохранена: {avg_gaze}")
            else:
                print(f"[Калибровка] Точка {self.current_point_index + 1}: недостаточно данных")
        
        # Переходим к следующей точке
        self.current_point_index += 1
        
        # Закрываем окно калибровки
        if self.calibration_window:
            self.calibration_window.close()
            self.calibration_window = None
        
        # Переходим к следующей точке или завершаем
        if self.current_point_index < len(self.calibration_points):
            self.root.after(500, self.next_calibration_step)
        else:
            self.root.after(500, self.finish_calibration)
    
    def finish_calibration(self):
        self.is_calibrating = False
        
        # Рассчитываем модель
        if self.calibrator.calibrate_from_points(self.calibration_data, 
                                                 (self.screen_width, self.screen_height)):
            self.is_calibrated = True
            self.status_label.config(text="Статус: Калибровка завершена ✓", fg='green')
            self.calibration_label.config(text="КАЛИБРОВКА ЗАВЕРШЕНА!", fg='green')
            print("[Калибровка] Успешно завершена!")
            
            # Разблокируем кнопки
            self.calibrate_btn.config(state='normal')
            self.tracking_btn.config(state='normal')
            
            # Показываем сообщение об успехе
            self.show_message("Калибровка завершена!", "green", 2000)
        else:
            self.status_label.config(text="Статус: Ошибка калибровки ✗", fg='red')
            self.calibration_label.config(text="ОШИБКА КАЛИБРОВКИ", fg='red')
            print("[Калибровка] Ошибка!")
            
            # Разблокируем кнопки
            self.calibrate_btn.config(state='normal')
            
            # Показываем сообщение об ошибке
            self.show_message("Ошибка калибровки. Попробуйте снова.", "red", 2000)
        
        self.root.after(2500, lambda: self.calibration_label.config(text=""))
    
    def show_message(self, text, color, duration=2000):
        """Показывает всплывающее сообщение"""
        msg = tk.Label(self.root, text=text,
                      font=("Arial", 24, "bold"), bg=color, fg='white')
        msg.place(relx=0.5, rely=0.5, anchor='center')
        self.root.after(duration, msg.destroy)
    
    def start_webcam(self):
        if self.is_running:
            return
        
        print("[Камера] Подключение...")
        self.current_camera = Webcam(cam_id=0)
        if not self.current_camera.connect():
            print("[Ошибка] Не удалось подключиться к камере")
            return
        
        self.is_running = True
        print("[Камера] Успешно подключена")
        self.update_frame()
    
    def display_frame(self, frame):
        """Отображает кадр в интерфейсе"""
        if frame is None:
            return
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def update_frame(self):
        """Основной цикл обновления кадра"""
        if not self.is_running:
            # Если камера не подключена, пробуем подключить через 1 секунду
            self.root.after(1000, self.update_frame)
            return
        
        frame = self.current_camera.read_frame()
        if frame is None or frame.size == 0:
            self.root.after(30, self.update_frame)
            return
        
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Рисуем трафарет для позиционирования во время калибровки
        if self.is_calibrating and self.current_point_index == 0:
            center_x, center_y = w // 2, h // 2
            face_width, face_height = int(w * 0.4), int(h * 0.5)
            
            cv2.rectangle(display_frame, 
                         (center_x - face_width//2, center_y - face_height//2),
                         (center_x + face_width//2, center_y + face_height//2),
                         (0, 255, 255), 2)
            
            eye_y = center_y - face_height//4
            eye_size = face_width // 10
            cv2.circle(display_frame, (center_x - face_width//4, eye_y), eye_size, (255, 255, 0), 2)
            cv2.circle(display_frame, (center_x + face_width//4, eye_y), eye_size, (255, 255, 0), 2)
            
            
            
        
        # Обработка взгляда (только если не в режиме сбора данных для калибровки)
        elif not self.is_calibrating or (self.is_calibrating and len(self.current_samples) > 0):
            gaze_offset = self.gaze_estimator.estimate_gaze(frame)
            self._last_gaze = gaze_offset
            
            if gaze_offset is not None:
                # Отображение точки взгляда на видео
                gaze_x = int((gaze_offset[0] + 1) * w / 2)
                gaze_y = int((gaze_offset[1] + 1) * h / 2)
                
                cv2.circle(display_frame, (gaze_x, gaze_y), 6, (0, 255, 0), -1)
                cv2.circle(display_frame, (gaze_x, gaze_y), 8, (255, 255, 0), 2)
                
                # Управление курсором если включено
                if self.tracking_enabled and self.is_calibrated:
                    # Добавляем в историю для сглаживания
                    self.gaze_history.append(gaze_offset)
                    if len(self.gaze_history) > 10:
                        self.gaze_history.pop(0)
                    
                    # Сглаживание
                    if len(self.gaze_history) >= 3:
                        smooth_gaze = np.mean(self.gaze_history[-3:], axis=0)
                    else:
                        smooth_gaze = gaze_offset
                    
                    screen_pos = self.calibrator.map_gaze_to_screen(
                        smooth_gaze, 
                        (self.screen_width, self.screen_height)
                    )
                    
                    if screen_pos:
                        # Плавное движение курсора
                        pyautogui.moveTo(screen_pos[0], screen_pos[1], duration=0.02)
                        
                        cv2.putText(display_frame, 
                                   f"Cursor: {screen_pos[0]}, {screen_pos[1]}",
                                   (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (255, 255, 255), 1)
                    
                    # Обработка кликов
                    self.eye_command.process(frame)
        
        # Отображаем статус
        status_text = "КАЛИБРАЦИЯ" if self.is_calibrating else \
                     "УПРАВЛЕНИЕ ВКЛ" if self.tracking_enabled else \
                     "УПРАВЛЕНИЕ ВЫКЛ"
        status_color = (0, 255, 255) if self.is_calibrating else \
                      (0, 255, 0) if self.tracking_enabled else \
                      (255, 0, 0)
       
        
        # Отображаем кадр
        self.display_frame(display_frame)
        
        # Продолжаем цикл
        self.root.after(30, self.update_frame)
    
    def cleanup_and_exit(self):
        """Очистка ресурсов и выход"""
        self.is_running = False
        if self.current_camera:
            self.current_camera.disconnect()
        self.root.destroy()
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.cleanup_and_exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackerApp(root)
    
    # Запускаем камеру при старте
    app.start_webcam()
    
    root.protocol("WM_DELETE_WINDOW", app.cleanup_and_exit)
    
    root.mainloop()