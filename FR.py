import cv2

import face_detection
import torch



class Face:
    def __init__(self, modelpath="model.h5", model_type="RetinaNetMobileNetV1", type="face"):
        self.points = []
        self.image = None
        self.name = None
        self.identification = 0
        self.model_path = modelpath
        self.model_type = model_type

    def detector(self, image=None, image_path=None, model_path=None, model_type=None):
        image_path = '2.jpg'
        if model_path is None:
            model_path = self.model_path
        if model_type is None:
            model_type = self.model_type
        if image is not None:
            self.image = image
        elif image_path is not None:
            image = cv2.imread(image_path)[:, :, ::-1]
            self.image = image

        self.detect()

    def detect(self, confidence_threshold=0.5, nsm_iou_threshold=0.3):
        detector = face_detection.build_detector(
            "RetinaNetResNet50", confidence_threshold=confidence_threshold, nms_iou_threshold=nsm_iou_threshold)
        print(face_detection.available_detectors)
        detections = detector.detect(self.image)
        for i in range(len(detections)):
            xmin, ymin, xmax, ymax = detections[i][0:4]
            self.points.append((xmin, ymin))
            self.points.append((xmax, ymax))



if __name__ == '__main__':
    print(torch.hub.DEFAULT_CACHE_DIR)
    print(torch.__version__)
    face = Face()
    face.detector(image_path="lenna.jpg")
    points = face.points
    print(points)
    print(torch.hub.get_dir())
