import face_detection


class FaceDetector:
    def __init__(self, model_type):
        if model_type == 'retina':
            self.detector = face_detection.build_detector(
                "RetinaNetResNet50", confidence_threshold=0.5, nms_iou_threshold=0.3)
        self.points = []
        self.score = []
        self.model_type = model_type
        self.image = None

    def detect(self, image):
        self.points = []
        self.score = []
        self.image = image
        detections = self.detector.detect(self.image)
        for i in range(len(detections)):
            xmin, ymin, xmax, ymax = detections[i][0:4]
            self.points.append((xmin, ymin))
            self.points.append((xmax, ymax))
            self.score.append(detections[i][4])


