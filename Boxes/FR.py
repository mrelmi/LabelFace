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


if __name__ == '__main__':
    import cv2
    import numpy as np

    detector = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=0.5, nms_iou_threshold=0.3)
    image = cv2.imread('2.jpg')
    images = np.expand_dims(image, axis=0)
    output = face_detection.RetinaNetResNet50.batched_detect_with_landmarks(detector, images)

    print(output)
    print(len(output[1][0]))
    for i in range(len(output[1][0])):
        for j in range(5):

            p1 = np.round(output[1][0][i][j][0])
            p2 = np.round(output[1][0][i][j][1])
            if j == 1:
                cv2.line(image, (np.round(output[1][0][i][j-1][0]), np.round(output[1][0][i][j-1][1])), (p1, p2),
                         (255, 255, 0))
            cv2.circle(image, (p1, p2), 4, (255, 0, 0))

    cv2.imshow("image", image)
    cv2.waitKey()
