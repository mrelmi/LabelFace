import face_detection
from skimage import transform as trans
import cv2
import numpy as np
import mxnet as mx
from torch.cuda import is_available as is_cuda_available


class ArcFace:
    def __init__(self, gpu_id, model_prefix, model_epoch):
        if gpu_id >= 0:
            ctx = mx.gpu(gpu_id)
        else:
            ctx = mx.cpu()
        image_size = (112, 112)
        self.model = self.get_model(ctx, image_size, model_prefix, model_epoch, 'fc1')

    def get_feature(self, inp):
        # a = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        inp = np.transpose(inp, (0, 3, 1, 2))
        data = mx.nd.array(inp)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        return self.model.get_outputs()[0].asnumpy()

    def get_model(self, ctx, image_size, prefix, epoch, layer):
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer + '_output']
        print(sym)
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        return model


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112):
    M, pose_index = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


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

    detector = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=0.5, nms_iou_threshold=0.3)
    image = cv2.imread('2.jpg')
    images = np.expand_dims(image, axis=0)
    output = face_detection.RetinaNetResNet50.batched_detect_with_landmarks(detector, images)

    # for i in range(len(output[1][0])):
    #     for j in range(5):
    #
    #         p1 = np.round(output[1][0][i][j][0])
    #         p2 = np.round(output[1][0][i][j][1])
    #         if j == 1:
    #             cv2.line(image, (np.round(output[1][0][i][j - 1][0]), np.round(output[1][0][i][j - 1][1])), (p1, p2),
    #                      (255, 255, 0))
    #         cv2.circle(image, (p1, p2), 4, (255, 0, 0))

    model_prefix = 'model-r100-ii/model'
    if is_cuda_available():
        device = 'cuda'
        gpuid = 0
    else:
        device = 'cpu'
        gpuid = -1
    arcface = ArcFace(gpu_id=gpuid, model_prefix=model_prefix, model_epoch=0)

    src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)
    src = np.expand_dims(src, axis=0)
    crops = []
    for i in range(len(output[1][0])):
        crop = norm_crop(image, output[1][0][i], )

        cv2.imshow("crop", crop)
        cv2.waitKey()
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(crop)
    crops = np.array(crops)
    embs = arcface.get_feature(crops)
