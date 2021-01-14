import os
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import traceback

from directories import FILES_DIR
from face_blocking import FaceBlocking
from model_types import ModelType


class App:
    def __init__(self, video_source=0, debug=False):
        self.DEFAULT_MIN_AGE = 0
        self.DEFAULT_MAX_AGE = 100
        self.video_source = video_source
        self.debug = debug
        self.debug_button = None
        self.vid = FaceBlocking(video_source=video_source)

        self.init_gui()

        self.pool(callback=lambda: self.draw_frame(), delay=15)

        self.window.mainloop()

    def init_gui(self):
        self.window = tk.Tk()
        self.window.title('Age Censorship')
        image = tk.PhotoImage(file="files/age_limit.gif")
        self.window.tk.call('wm', 'iconphoto', self.window._w, image)

        self.canvas = tk.Canvas(self.window, width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=0, columnspan=3)

        self.build_age_restriction_area()

        self.build_options_area()

    def build_options_area(self):
        options_area = tk.LabelFrame(self.window, text="Options")
        options_area.grid(row=3, rowspan=4, columnspan=3, sticky="nsew", padx=5, pady=5)

        self.build_age_estimation_model_area(options_area)

        face_detection_model_frame = tk.LabelFrame(options_area, text="Face detection model")
        face_detection_model_frame.pack(expand=True, padx=5, pady=5)

        self.chosen_detection_model_type = tk.StringVar()
        radiobutton1 = tk.Radiobutton(face_detection_model_frame,
                                      text='Haar Cascade',
                                      variable=self.chosen_detection_model_type,
                                      value='base',
                                      command=self.set_detection_type)
        radiobutton1.pack(expand=True, side=tk.LEFT, padx=5, pady=5)
        radiobutton1.select()

        radiobutton2 = tk.Radiobutton(face_detection_model_frame,
                                      text='ResNet10',
                                      variable=self.chosen_detection_model_type,
                                      value='caffe',
                                      command=self.set_detection_type)
        radiobutton2.pack(expand=True, side=tk.LEFT, padx=5, pady=5)

        self.build_debug_mode_area(options_area)

    def build_age_estimation_model_area(self, options_area):
        age_estimation_model_frame = tk.LabelFrame(options_area, text="Age estimation model")
        age_estimation_model_frame.pack(expand=True, padx=5, pady=5)

        self.chosen_model_type = tk.IntVar()
        radiobutton1 = tk.Radiobutton(age_estimation_model_frame,
                                      text=ModelType.WideResNet.name,
                                      variable=self.chosen_model_type,
                                      value=ModelType.WideResNet.value,
                                      command=self.set_age_estimation_method)
        radiobutton1.pack(expand=True, side=tk.LEFT, padx=5, pady=5)
        radiobutton1.select()

        radiobutton2 = tk.Radiobutton(age_estimation_model_frame,
                                      text=ModelType.InceptionResNetV2.name,
                                      variable=self.chosen_model_type,
                                      value=ModelType.InceptionResNetV2.value,
                                      command=self.set_age_estimation_method)
        radiobutton2.pack(expand=True, side=tk.LEFT, padx=5, pady=5)

    def build_debug_mode_area(self, options_area):
        debug_mode_frame = tk.Frame(options_area)
        debug_mode_frame.pack(expand=True, padx=5, pady=5)

        tk.Label(debug_mode_frame, text="Debug mode:").pack(expand=True, side=tk.LEFT, padx=5, pady=5)

        self.debug_button = tk.Button(debug_mode_frame, text="Off", width=12, command=self.toggle)
        self.debug_button.pack(expand=True, side=tk.LEFT, padx=5, pady=5)

    def build_age_restriction_area(self):
        self.age_restriction_area = tk.LabelFrame(self.window, text="Age restrictions")
        self.age_restriction_area.grid(row=1, rowspan=2, columnspan=3, sticky="nsew", padx=5, pady=5)

        right_frame = tk.Frame(self.age_restriction_area)
        right_frame.pack(expand=True, side=tk.LEFT, padx=5, pady=5)
        left_frame = tk.Frame(self.age_restriction_area)
        left_frame.pack(expand=True, side=tk.LEFT, padx=5, pady=5)

        tk.Label(right_frame, text="Minimal age:").pack(padx=5)
        self.minimal_age_button = tk.Entry(right_frame)
        self.minimal_age_button.pack(padx=5, pady=5)

        tk.Label(left_frame, text="Maximal age:").pack(padx=5)
        self.maximal_age_button = tk.Entry(left_frame)
        self.maximal_age_button.pack(padx=5, pady=5)

    def toggle(self):
        if self.debug_button is not None:
            button_state_ = self.debug_button.config('text')[-1] == "Off"

            if button_state_ is True:
                self.debug_button.config(text='On')
            else:
                self.debug_button.config(text='Off')
            self.debug = button_state_

    def draw_frame(self):
        try:
            frame = self.vid.get_processed_frame(self.get_age_restriction_values(), _debug=self.debug)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        except Exception:
            print(traceback.format_exc())

    def pool(self, callback, delay):
        callback()
        self.window.after(delay, lambda: self.pool(callback, delay))

    def get_age_restriction_values(self):
        minimal_age = self.DEFAULT_MIN_AGE
        maximal_age = self.DEFAULT_MAX_AGE

        minimal_age_button_value = self.minimal_age_button.get()
        if minimal_age_button_value != "":
            try:
                minimal_age = int(minimal_age_button_value)
            except Exception:
                print(traceback.format_exc())

        maximal_age_button_value = self.maximal_age_button.get()
        if maximal_age_button_value != "":
            try:
                maximal_age = int(self.maximal_age_button.get())
            except Exception:
                print(traceback.format_exc())

        return minimal_age, maximal_age

    def set_detection_type(self):
        self.vid.set_detection_type(self.chosen_detection_model_type.get())

    def set_age_estimation_method(self):
        self.vid.set_age_estimation_model(ModelType(self.chosen_model_type.get()))
