import tkinter as tk
import cv2
from PIL import Image, ImageTk
from camera.webcam import Webcam
from system.gaze_estimator import GazeEstimator
import json
import os


class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()

        root.title("Eye Tracker — Калибровка")
        root.attributes('-fullscreen', True)
        root.configure(bg='black')
        root.bind('<Escape>', lambda e: root.destroy())

        self.video_label = tk.Label(root, bg='black')
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.overlay = tk.Canvas(root, bg='', highlightthickness=0)
        self.overlay.place(x=0, y=0, relwidth=1, relheight=1)

        self.current_camera = None
        self.is_running = False
        self.gaze_estimator = GazeEstimator()
        self._last_gaze = None
        self.gaze_samples = []

        self.start_camera()
        root.after(1000, self.show_alignment_ui)

    def start_camera(self):
        camera = Webcam(cam_id=0)
        if camera.connect():
            self.current_camera = camera
            self.is_running = True
            self._update_frame()
        else:
            print("[Ошибка] Не удалось подключить вебкамеру")

    def show_alignment_ui(self):
        cx, cy = self.screen_width // 2, self.screen_height // 2

        self.overlay.create_oval(
            cx - 120, cy - 160,
            cx + 120, cy + 160,
            outline='white', width=2, dash=(10, 5)
        )
        self.overlay.create_text(cx, cy - 180, text="Расположите лицо в овал", fill='white', font=("Arial", 16))

        self.ready_btn = tk.Button(
            self.root, text="Готов", font=("Arial", 16),
            command=self.on_ready_clicked, bg='green', fg='white'
        )
        self.ready_btn.place(relx=0.5, rely=0.9, anchor='center')

    def on_ready_clicked(self):
        self.ready_btn.destroy()
        self.overlay.delete("all")

        self.calibration_dot = tk.Label(self.root, bg='red', width=2, height=1)
        self.calibration_dot.place(
            x=self.screen_width // 2 - 10,
            y=self.screen_height // 2 - 10
        )

        self.countdown_label = tk.Label(
            self.root, text="3", bg='black', fg='white', font=("Arial", 24)
        )
        self.countdown_label.place(relx=0.5, y=50, anchor='center')

        self.gaze_samples.clear()
        self.countdown(3)

    def countdown(self, sec):
        if sec > 0:
            self.countdown_label.config(text=str(sec))
            self.root.after(1000, self.countdown, sec - 1)
        else:
            self.finish_calibration()

    def finish_calibration(self):
        self.calibration_dot.destroy()
        self.countdown_label.destroy()

        if self.gaze_samples:
            avg_x = sum(s[0] for s in self.gaze_samples) / len(self.gaze_samples)
            avg_y = sum(s[1] for s in self.gaze_samples) / len(self.gaze_samples)
            offset = (avg_x, avg_y)

            calib_data = {
                "bias_x": offset[0],
                "bias_y": offset[1],
                "screen_width": self.screen_width,
                "screen_height": self.screen_height
            }
            with open("calibration.json", "w") as f:
                json.dump(calib_data, f)
            print(f"[Калибровка] Сохранено: {offset}")

        self.root.after(1000, self.root.destroy)

    def _update_frame(self):
        if not self.is_running:
            return

        frame = self.current_camera.read_frame()
        if frame is not None and frame.size > 0:
            gaze_offset = self.gaze_estimator.estimate_gaze(frame)
            self._last_gaze = gaze_offset

            if hasattr(self, 'calibration_dot') and gaze_offset is not None:
                self.gaze_samples.append(gaze_offset)

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            except Exception:
                pass

        self.root.after(30, self._update_frame)

    def __del__(self):
        if hasattr(self, 'current_camera') and self.current_camera:
            self.current_camera.disconnect()


if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()