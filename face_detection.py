import os
import cv2
import numpy as np

from directories import MODELS_DIR


class FaceDetector:
    def __init__(self):
        self._roi = None
        self._label = None
        self._color_roi = None
        self._detection_type = 'caffe'

    def get_coordinates(self, _img_array, _multi_face=False):
        if self._detection_type == "base":
            _img_array = cv2.cvtColor(_img_array, cv2.COLOR_BGR2GRAY)
            return self.get_base_coordinates(_img_array)
        elif self._detection_type == "caffe":
            return self.get_caffe_coordinates(_img_array, _multi_face)

    def face_detection(self, _img_array, _img_path, _label, _multi_face=False):
        if self._detection_type == "base":
            self._roi, self._label = self.base_detection(_img_array, _label)
        elif self._detection_type == "caffe":
            _img_array_ = cv2.imread(_img_path)
            self._roi, self._label, self._color_roi = self.caffe_detection(_img_array_, _label, _multi_face)
        return self._roi, self._label, self._color_roi

    def caffe_detection(self, _img_array, _label, _multi_face):
        start_x, start_y, end_x, end_y = self.get_caffe_coordinates(_img_array, _multi_face)[0]
        mono_img_array = cv2.cvtColor(_img_array, cv2.COLOR_BGR2GRAY)
        _roi = mono_img_array[start_y:end_y, start_x:end_x]
        _color_roi = _img_array[start_y:end_y, start_x:end_x]
        return _roi, _label, _color_roi

    @staticmethod
    def get_caffe_coordinates(_img_array, _multi_face=False):
        def get_best_faces(multi_face):
            confidences = []
            for j in range(0, detections.shape[2]):
                conf = detections[0, 0, j, 2]
                if conf > 0.9:
                    confidences.append(conf)
            if confidences:
                if multi_face:
                    return confidences
                return [max(confidences)]
            return []

        prototxt = os.path.join(MODELS_DIR, "deploy.prototxt")
        model = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        # load image and construct an input blob for the image by resizing to a fixed 300x300 pixels and then normalizing it
        (h, w) = _img_array.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(_img_array, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        best_conf = get_best_faces(_multi_face)
        faces = []
        if not best_conf:
            return
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence in best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x_, start_y_, end_x_, end_y_) = box.astype("int")
                if len(best_conf) == 1:
                    return [(start_x_, start_y_, end_x_, end_y_)]
                else:
                    faces.append((start_x_, start_y_, end_x_, end_y_))
        return faces

    @staticmethod
    def get_base_coordinates(_img_array, _multi_face=True):
        # find faces on image
        cascade_clf = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        faces = cascade_clf.detectMultiScale(_img_array, scaleFactor=1.5, minNeighbors=5)
        if isinstance(faces, tuple):
            return
        if _multi_face:
            return list(map(lambda face: (face[0], face[1], face[0] + face[2], face[1] + face[3]), faces))
        return faces[0][0], faces[0][1], faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]

    def base_detection(self, _img_array, _label):
        start_x, start_y, end_x, end_y = self.get_base_coordinates(_img_array)
        _roi = _img_array[start_y:end_y, start_x:end_x]
        return _roi, _label

    def set_detection_type(self, detection_type):
        self._detection_type = detection_type
        print("Detection type: " + self._detection_type)
