import json
import os
import shutil

with open("../annotations/new-train.json", "r") as f:
    d = json.loads(f.read())
with open("../annotations/train.json", "r") as f:
    d2 = json.loads(f.read())


mpd_ids = {}

new_images = []

for img in d["images"]:
    mpd_ids[img["id"]] = 100_000 + len(new_images)
    img["id"] = mpd_ids[img["id"]]
    new_images.append(img)
new_images.extend(d2["images"])

new_annots = []
for annot in d["annotations"]:
    annot["id"] = annot["image_id"] = mpd_ids[annot["image_id"]]
    new_annots.append(annot)
new_annots.extend(d2["annotations"])

new_images.sort(key=lambda img: img["id"])
new_annots.sort(key=lambda annot: annot["id"])

d["annotations"] = new_annots
d["images"] = new_images
with open("../annotations/new-train2.json", "w") as f:
    f.write(json.dumps(d, indent=4))
