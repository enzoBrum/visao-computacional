from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from os import cpu_count
from pathlib import Path
import re
from traceback import print_exc

import cv2
import tqdm

from inference import Hrnetv2_w18_dark, Yolov5_Lite

YOLO_NET: Yolov5_Lite = None
HR_NET: Hrnetv2_w18_dark = None


def init():
    global YOLO_NET
    global HR_NET
    YOLO_NET = Yolov5_Lite(
        "./v5lite-finetuned-c.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    HR_NET = Hrnetv2_w18_dark(
        "./hrnet-2.onnx",
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        confidence_threshold=0.4,
    )


def process_roi(src: Path, dst: Path):
    img = YOLO_NET.detect(cv2.imread(str(src)))[0]
    roi = HR_NET.detect(img)

    dst.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(dst), cv2.resize(roi, (228, 228)))
    return src, dst


DATASET = Path("./dataset/")
DST = Path("./extracted-rois")
DST.mkdir(exist_ok=True)
IMAGES: list[Path] = []
for ext in ("jpg", "JPG", "tiff"):
    IMAGES.extend(DATASET.glob(f"**/*.{ext}"))


invalid_image = re.compile(r".*SMPD.*_(RF|RP|P)_.*")

with ProcessPoolExecutor(cpu_count(), initializer=init) as pool:
    futures: list[Future] = []

    for img in IMAGES:
        if invalid_image.match(str(img)):
            continue

        dirname = img.parts[1]
        for p in img.parts[2:-1]:
            dirname += f"-{p}"

        dirpath = DST / dirname
        dst = dirpath / img.name

        futures.append(pool.submit(process_roi, src=img, dst=dst))

    for fut in tqdm.tqdm(as_completed(futures)):
        try:
            src, dst = fut.result()
        except:
            pass

    num_to_img = defaultdict(int)
    for dirpath in DST.iterdir():
        num_imgs = len(list(dirpath.iterdir()))
        if num_imgs == 0:
            print(f"RM EMPTY: {dirpath}")
            dirpath.rmdir()

        num_to_img[num_imgs] += 1
    print("\n".join(f"{k}: {v}" for k, v in num_to_img.items()))
