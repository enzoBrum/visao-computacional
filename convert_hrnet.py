from collections import defaultdict
import json

with open("./hrnet-final.json") as f:
    d = json.loads(f.read())

d2 = defaultdict(list)
for k, v in d.items():
    d2[round(float(k), 1)].extend(v)

with open("./hrnet-conv.json", "w") as f:
    f.write(json.dumps(d2, sort_keys=True))
