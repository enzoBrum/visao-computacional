import json
import math
import random

import re

rgx = re.compile(r".*SMPD.*_(RF|RP|P)_.*")

with open("./coco-final.json") as f:
    d = json.loads(f.read())
with open("./hrnet-final.json") as f:
    d2 = json.loads(f.read())




id_to_img = {img["id"]: img for img in d["images"]}
annot_kp_to_img = {
    id_to_img[annot["image_id"] - 1]["file_name"]
    for annot in d["annotations"]
    if "keypoints" in annot
}
bad_bad = {x.replace("/", "-") for x in d2["0.0"]}
imgs_too_close = []

for annot in d["annotations"]:
    if "keypoints" not in annot:
        continue

    x1, y1, _, x2, y2, _ = annot["keypoints"]

    if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < 5 and not rgx.match(id_to_img[annot["image_id"] - 1]["file_name"]):
        imgs_too_close.append(id_to_img[annot["image_id"] - 1]["file_name"])
        
bad_bad = [
            img
            for img in bad_bad if not rgx.match(img)
        ]

d3 = d2.copy()
for k in ("0.0", "0.1", "0.2", "0.3"):
    d3.pop(k)
print(
    {k: random.choice(list(v2 for v2 in v if not rgx.match(v2))) for k, v in d2.items()},
    {k: len(v) for k, v in sorted(d2.items())},
    sum([len(v) for k, v in sorted(d3.items())]),
    len(imgs_too_close) + len(bad_bad),
    len(
        bad_bad
    ),
    '\n'.join(random.sample(imgs_too_close, min(len(imgs_too_close), 5))),
    '\n'.join(random.sample(bad_bad, 5)),
)
