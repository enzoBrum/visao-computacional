import math
from sys import argv

import cv2
from cv2.typing import MatLike
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort
from torchvision import transforms as v2

from numpy.typing import NDArray
import onnxruntime as ort
from tempfile import TemporaryDirectory

from torchvision.transforms import v2
from PIL import Image
from sys import argv

class Yolov5_Lite:
    """
    FROM: https://github.com/ppogg/YOLOv5-Lite/blob/9d649a64f8293b128c672adcd09aa762df9bc6bc/python_demo/onnxruntime/v5lite.py
    """

    net: ort.InferenceSession
    confThreshold: float
    nmsThreshold: float
    input_shape: tuple[int, int]

    def __init__(
        self,
        model_path: str,
        providers: list[str] | None = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> None:
        if providers is None:
            providers = ort.get_available_providers()
            if "TensorrtExecutionProvider" in providers:
                providers.remove("TensorrtExecutionProvider")

        self.net = ort.InferenceSession(model_path, providers=providers)
        self.confThreshold = conf_threshold
        self.nmsThreshold = nms_threshold
        self.classes = {0: "Palm"}

        w, h = self.net.get_inputs()[0].shape[2:4]
        self.input_shape = (w, h)
        print(f"YOLO: {self.input_shape}")

    def letterBox(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    left,
                    self.input_shape[1] - neww - left,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )  # add border
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    self.input_shape[0] - newh - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def postprocess(self, frame, outs, pad_hw):
        newh, neww, padh, padw = pad_hw
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[4]
            if scores.any() > self.confThreshold:
                x1 = int((detection[0] - padw) * ratiow)
                y1 = int((detection[1] - padh) * ratioh)
                x2 = int((detection[2] - padw) * ratiow)
                y2 = int((detection[3] - padh) * ratioh)
                classIds.append(0)
                confidences.append(scores)
                boxes.append([x1, y1, x2, y2])
            else:
                raise Exception(f"Could not detect box. Scores: {scores}")

        # # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # # lower confidences.
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confThreshold, self.nmsThreshold
        )

        bbs = []
        for ind in indices:
            bbs.append(boxes[ind][0:4])
        return bbs

    def drawPred(self, frame, classId, conf, x1, y1, x2, y2):
        # Draw a bounding box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

        label = "%.2f" % conf
        text = "%s:%s" % (self.classes[int(classId)], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1 = max(y1, labelSize[1])
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            (0, 255, 0),
            thickness=1,
        )
        return frame

    def detect(self, srcimg):
        img, newh, neww, top, left = self.letterBox(srcimg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        print(f"YOLO BLOB: {blob.shape}")
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0]
        print(f"YOLO outs: {outs.shape}")
        print(outs)
        bbs = self.postprocess(srcimg, outs, (newh, neww, top, left))

        imgs = []
        h, w = srcimg.shape[0:2]
        for bb in bbs:
            x1, y1, x2, y2 = bb

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            imgs.append(srcimg[y1:y2, x1:x2])
        return imgs


class Hrnetv2_w18_dark:
    """
    inspired from: https://github.com/open-mmlab/mmpose/blob/71ec36ebd63c475ab589afc817868e749a61491f/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256.py
    """

    net: ort.InferenceSession

    std = np.array([58.395, 57.12, 57.375], np.float16)
    mean = np.array([123.675, 116.28, 103.53], np.float16)

    confidence_threshold: float

    def __init__(
        self,
        model_path: str,
        providers: list[str] | None = None,
        confidence_threshold: float = 0.0,
    ) -> None:
        if providers is None:
            providers = ort.get_available_providers()
            if "TensorrtExecutionProvider" in providers:
                providers.remove("TensorrtExecutionProvider")
        self.net = ort.InferenceSession(model_path, providers=providers)
        w, h = self.net.get_inputs()[0].shape[2:4]
        self.input_shape = (w, h)
        self.confidence_threshold = confidence_threshold
        print(f"HRNET shape: {self.input_shape}")

    def preprocess(self, img: NDArray) -> NDArray[np.float32]:
        img = cv2.resize(img, self.input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)

        img = (img - self.mean) / self.std

        return np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2)).astype(
            np.float16
        )

    def get_keypoints(
        self, result: NDArray[np.float16]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        heatmaps = result[0][0]
        num_keypoints = 2

        H = W = 64
        multiplier = 4

        points = []
        last = None
        for i in range(num_keypoints):
            heatmap = heatmaps[i].astype(np.float32)

            if last is None:
                y, x = np.unravel_index(np.argmax(heatmap), (H, W))
                last = (x, y)
            else:
                flat = heatmap.flatten()
                indexes = np.argsort(flat)
                r, c = np.unravel_index(indexes, heatmap.shape)

                for ii, jj in list(zip(r, c))[::-1]:
                    if math.sqrt((jj - last[0]) ** 2 + (ii - last[1]) ** 2) > 10:
                        break
                x = jj
                y = ii

            score = heatmap[y, x]

            if score < self.confidence_threshold:
                raise Exception(f"Could not detect keypoints: {score}")

            dx = 0.0
            dy = 0.0

            if 0 < x < W - 1:
                diff_x = heatmap[y, x + 1] - heatmap[y, x - 1]
                dx = np.sign(diff_x) * 0.25

            if 0 < y < H - 1:
                diff_y = heatmap[y + 1, x] - heatmap[y - 1, x]
                dy = np.sign(diff_y) * 0.25

            final_x = round((x + dx) * multiplier)
            final_y = round((y + dy) * multiplier)

            points.append((final_x, final_y))

        return points

    def get_rotation_matrix(
        self, p1: tuple[int, int], p2: tuple[int, int], center: tuple[int, int]
    ) -> MatLike:
        x1, y1 = p1
        x2, y2 = p2
        angle = math.atan2(y2 - y1, x2 - x1)

        return cv2.getRotationMatrix2D(center, math.degrees(angle), 1.0)

    def postprocess(
        self, image: MatLike, keypoints: tuple[tuple[int, int], tuple[int, int]]
    ) -> MatLike:

        image = cv2.resize(image, self.input_shape)

        w, h = self.input_shape
        center = (w // 2, h // 2)
        rotation_matrix = self.get_rotation_matrix(keypoints[0], keypoints[1], center)

        _, y = np.dot(rotation_matrix, np.array([keypoints[0][0], keypoints[0][1], 1]))

        if y > self.input_shape[1] // 2:
            rotation_matrix = self.get_rotation_matrix(
                keypoints[1], keypoints[0], center
            )

        image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)

        kp_x1, kp_y1 = np.dot(
            rotation_matrix, np.array([keypoints[0][0], keypoints[0][1], 1])
        )
        kp_x2, kp_y2 = np.dot(
            rotation_matrix, np.array([keypoints[1][0], keypoints[1][1], 1])
        )

        kp_x1 = round(kp_x1)
        kp_x2 = round(kp_x2)
        kp_y1 = round(kp_y1)
        kp_y2 = round(kp_y2)

        midpoint_x = (kp_x1 + kp_x2) // 2
        midpoint_y = (kp_y1 + kp_y2) // 2

        rect_center_y = midpoint_y + 0.8 * abs(kp_x1 - kp_x2)
        rect_side = abs(kp_x1 - kp_x2) * 1.2

        rect_x1 = round(midpoint_x - rect_side // 2)
        rect_x2 = round(midpoint_x + rect_side // 2)

        rect_y1 = round(rect_center_y - rect_side // 2)
        rect_y2 = round(rect_center_y + rect_side // 2)

        roi = image[rect_y1:rect_y2, rect_x1:rect_x2]
        return cv2.resize(roi, (228, 228))

    def detect(self, image: MatLike):
        # see: https://github.com/open-mmlab/mmpose/issues/949
        img = self.preprocess(image)
        print(f"HRNET: {img.shape}")
        result = self.net.run(None, {self.net.get_inputs()[0].name: img})
        print(f"HRNET result: {result[0].shape}")
        keypoints = self.get_keypoints(result)

        return self.postprocess(image, keypoints)





class FeatureExtractor:
    net: ort.InferenceSession
    rec_threshold: float
    val_transforms = v2.Compose(
        [v2.ToTensor(), v2.Normalize([0.485, 0.485, 0.406], [0.229, 0.224, 0.225])]
    )

    def __init__(
        self,
        model_path: str,
        providers: list[str] | None = None,
        rec_threshold: float = 0.3,
    ) -> None:
        if providers is None:
            providers = ort.get_available_providers()
            if "TensorrtExecutionProvider" in providers:
                providers.remove("TensorrtExecutionProvider")
        self.net = ort.InferenceSession(model_path, providers=providers)
        self.rec_threshold = rec_threshold

    def get_embbeddings(self, imgs: list[str]) -> NDArray[np.float32]:
        images = []
        for im in imgs:
            images.append(self.val_transforms(Image.open(im)).numpy())

        print(f"FT: {np.array(images).shape}")
        r= self.net.run(None, {"in": images})
        print(f"FT R.: {r[0].shape}")
        return r

    def normalize(self, x, axis=-1, eps=1e-8):
        norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
        return x / (norm + eps)

    def is_same_person(
        self, im1: NDArray[np.float32], im2: NDArray[np.float32]
    ) -> bool:
        print(np.dot(self.normalize(im1), self.normalize(im2)) )
        return np.dot(self.normalize(im1), self.normalize(im2)) >= self.rec_threshold


if __name__ == "__main__":
    providers = ["CPUExecutionProvider"]
    yolo = Yolov5_Lite("./v5lite-finetuned-c.onnx", providers)
    hrnet = Hrnetv2_w18_dark("./hrnet.onnx", providers)
    feature_extractor = FeatureExtractor("./feature_extractor.onnx", providers)
    print(hrnet.net.get_inputs()[0].name)
    print(feature_extractor.net.get_inputs()[0].name)
    print(feature_extractor.net.get_outputs()[0].name)

    im1 = "/home/erb/Downloads/IMG_8210.jpg"
    im2 = "/home/erb/Downloads/IMG_8210.jpg"

    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)

    palm1 = yolo.detect(im1)[0]
    palm2 = yolo.detect(im2)[0]

    roi1 = hrnet.detect(palm1)
    roi2 = hrnet.detect(palm2)

    with TemporaryDirectory() as dirpath:
        roi1 = cv2.imwrite(f"{dirpath}/im1.jpg", roi1)
        roi2 = cv2.imwrite(f"{dirpath}/im2.jpg", roi2)

        emb1, emb2 = feature_extractor.get_embbeddings([f"{dirpath}/im1.jpg", f"{dirpath}/im2.jpg"])[0]
        print(emb1)
        print(feature_extractor.is_same_person(emb1, emb2))


