import os
import cv2
import numpy as np

from keras.models import load_model

from directories import MODELS_DIR
from model_types import ModelType



class AgeEstimator:

    def __init__(self):
        self.models = {
            ModelType.WideResNet: load_model(os.path.join(MODELS_DIR, "WideResNet.hdf5")),
            ModelType.InceptionResNetV2: load_model(os.path.join(MODELS_DIR, "InceptionResNetV2.hdf5")),
        }
        self.switch_model()

    def switch_model(self, model_type=ModelType.WideResNet):
        self.active_model = self.models[model_type]
        print("Active model: " + model_type.name)

    def estimate(self, face):
        resized_face = np.expand_dims(cv2.resize(face, (64, 64), 1, 1), axis=0)
        return self.active_model.predict(resized_face)
