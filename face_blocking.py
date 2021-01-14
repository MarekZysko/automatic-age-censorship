import cv2
import numpy as np

from age_estimation import AgeEstimator
from face_detection import FaceDetector



class FaceBlocking:
    def __init__(self, video_source=0):
        self.COVER_COLOR = (0, 0, 0)

        self.detector = FaceDetector()
        self.age_estimator = AgeEstimator()

        self.capture = cv2.VideoCapture(video_source)
        if not self.capture.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.capture.isOpened():
            self.capture.release()

    def get_processed_frame(self, _age_restrictions=(), _debug=False):
        val, frame = self.capture.read()

        faces = self.detector.get_coordinates(frame, _multi_face=True)
        faces = faces if faces is not None else []
        for i in range(len(faces)):
            start_x, start_y, end_x, end_y = faces[i]
            roi_color = frame[start_y:end_y, start_x:end_x]
            pred_gender, pred_age = self.age_estimator.estimate(roi_color)
            estimated_age = np.argmax(pred_age)
            if not _age_restrictions[0] <= estimated_age <= _age_restrictions[1]:
                self.block_face(frame, start_x, start_y, end_x, end_y)
            if _debug:
                self.draw_debug(frame, estimated_age, start_x, start_y, end_x, end_y)

        return frame

    def draw_debug(self, _frame, _age, _start_x, _start_y, _end_x, _end_y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)
        stroke = 2
        cv2.putText(_frame, "Age: " + str(_age), (_start_x, _start_y), font, 1, color, stroke, cv2.LINE_AA)
        cv2.rectangle(_frame, (_start_x, _start_y), (_end_x, _end_y), color, stroke)

    def block_face(self, _frame, _start_x, _start_y, _end_x, _end_y):
        cv2.rectangle(_frame, (_start_x, _start_y), (_end_x, _end_y), self.COVER_COLOR, -1)

    def set_detection_type(self, detection_type):
        self.detector.set_detection_type(detection_type)

    def set_age_estimation_model(self, estimation_model):
        self.age_estimator.switch_model(estimation_model)
