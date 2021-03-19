from Boxes.FR import FaceDetector
import cv2
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

class BoxRecommender:
    def __init__(self, modelpath="model.h5", box_type="face"):
        self.points = []
        self.image = None
        self.name = None
        self.identification = 0
        self.model_path = modelpath
        if box_type == "face":
            self.detector = FaceDetector("retina")

    def detect(self, image=None, image_path=None):
        if image is not None:
            self.image = image
        else:
            image = Image.open(image_path)
            image = np.array(image)[:, :, ::-1]
            self.image = image
        self.detector.detect(self.image)
        self.points = self.detector.points
