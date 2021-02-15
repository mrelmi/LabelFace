from Boxes.FR import FaceDetector
import cv2


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
        if self.image is not None:
            self.image = image
        else:
            image = cv2.imread(image_path)[:, :, ::-1]
            self.image = image
        self.detector.detect(self.image)
        self.points = self.detector.points



