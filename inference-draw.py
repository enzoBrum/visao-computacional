import math
from sys import argv

import cv2
from cv2.typing import MatLike
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort


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

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0]
        bbs = self.postprocess(srcimg, outs, (newh, neww, top, left))

        imgs = []
        h, w = srcimg.shape[0:2]
        final_bbs = []
        for bb in bbs:
            x1, y1, x2, y2 = bb

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            final_bbs.append((x1, y1, x2, y2))
            imgs.append(srcimg[y1:y2, x1:x2])
        return imgs, final_bbs


class Hrnetv2_w18_dark:
    """
    Hrnet from: https://github.com/open-mmlab/mmpose/blob/71ec36ebd63c475ab589afc817868e749a61491f/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256.py
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

            print(score)
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

            # img = cv2.circle(
            #    img, (final_x, final_y), radius=3, color=(0, 255, 0), thickness=-1
            # )
        print(f"KEYPOINS: {points}")
        return points

    def get_rotation_matrix(
        self, p1: tuple[int, int], p2: tuple[int, int], center: tuple[int, int]
    ) -> MatLike:
        x1, y1 = p1
        x2, y2 = p2
        angle = math.atan2(y2 - y1, x2 - x1)
        # angle = math.atan(abs(y1 - y2) / abs(x1 - x2))

        return cv2.getRotationMatrix2D(center, math.degrees(angle), 1.0)

    def postprocess(
        self, image: MatLike, keypoints: tuple[tuple[int, int], tuple[int, int]]
    ) -> MatLike:

        image = cv2.resize(image, self.input_shape)

        w, h = self.input_shape
        center = (w // 2, h // 2)
        rotation_matrix = self.get_rotation_matrix(keypoints[0], keypoints[1], center)

        x, y = np.dot(rotation_matrix, np.array([keypoints[0][0], keypoints[0][1], 1]))

        if y > self.input_shape[1] // 2:
            rotation_matrix = self.get_rotation_matrix(
                keypoints[1], keypoints[0], center
            )

        image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
        colors = [(255, 255, 0), (255, 0, 255)]

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

        rect_x1 = max(round(midpoint_x - rect_side // 2), 0)
        rect_x2 = max(round(midpoint_x + rect_side // 2), 0)

        rect_y1 = min(round(rect_center_y - rect_side // 2), 255)
        rect_y2 = min(round(rect_center_y + rect_side // 2), 255)
        
        print(rect_x1, rect_y1, rect_x2, rect_y2)

        crop = image[rect_y1:rect_y2, rect_x1:rect_x2].copy()

        image = cv2.rectangle(
            image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255)
        )
        image = cv2.circle(image, (midpoint_x, midpoint_y), 3, (255, 128, 60), 3)
        image = cv2.circle(
            image, (midpoint_x, round(rect_center_y)), 3, (255, 60, 128), 3
        )

        for i in range(2):
            arr = np.array([keypoints[i][0], keypoints[i][1], 1])
            arr = np.dot(rotation_matrix, arr)
            #
            x = int(arr[0])
            y = int(arr[1])
            # x, y = keypoints[i]
            color = colors[i]
            image = cv2.circle(image, (x, y), 3, color, 3)
        return image, keypoints, crop

    def detect(self, image: MatLike):
        # see: https://github.com/open-mmlab/mmpose/issues/949
        img = self.preprocess(image)
        result = self.net.run(None, {self.net.get_inputs()[0].name: img})
        keypoints = self.get_keypoints(result)

        return self.postprocess(image, keypoints)


if __name__ == "__main__":
    matplotlib.use("TkAgg")

    img_path = argv[1]
    img = cv2.imread(img_path)
    img_og = img.copy()
    net = Yolov5_Lite("./v5lite-finetuned-c.onnx", ["CPUExecutionProvider"])

    imgs, bbs = net.detect(img.copy())

    hrnet = Hrnetv2_w18_dark("./hrnet-2.onnx", ["CPUExecutionProvider"])
    roi, keypoints, crop = hrnet.detect(imgs[0])

    kp_x1, kp_y1 = keypoints[0]
    kp_x2, kp_y2 = keypoints[1]

    angle = math.degrees(math.atan2(kp_y2 - kp_y1, kp_x2 - kp_x1))

    rotation_matrix_kps = cv2.getRotationMatrix2D((256 // 2, 256 // 2), angle, 1.0)

    # kp_x1, kp_y1 = np.dot(rotation_matrix_kps, np.array([kp_x1, kp_y1, 1]))
    # kp_x2, kp_y2 = np.dot(rotation_matrix_kps, np.array([kp_x2, kp_y2, 1]))

    h, w = img.shape[:2]

    bbox = bbs[0]
    bb_x1, bb_y1, bb_x2, bb_y2 = bbs[0]

    bb_center = ((bb_x1 + bb_x2) // 2, (bb_y1 + bb_y2) // 2)
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    rect = (bb_center, (bbox_w, bbox_h), angle)
    box = np.intp(cv2.boxPoints(rect))

    kp_x1 = round(kp_x1 * (bbox_w / 256)) + bbox[0]
    kp_x2 = round(kp_x2 * (bbox_w / 256)) + bbox[0]
    kp_y1 = round(kp_y1 * (bbox_h / 256)) + bbox[1]
    kp_y2 = round(kp_y2 * (bbox_h / 256)) + bbox[1]

    result = cv2.rectangle(img, bbox[:2], bbox[2:], (0, 0, 255), 24)
    result = cv2.circle(result, (kp_x1, kp_y1), 45, (255, 255, 0), 45)
    result = cv2.circle(result, (kp_x2, kp_y2), 45, (255, 0, 255), 45)

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Pontos-chave e box")
    plt.axis("off")

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title("ROI")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    plt.title("CROPPED")
    plt.axis("off")

    plt.show()
