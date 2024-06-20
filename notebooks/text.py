from PIL import Image
import os
os.environ['USE_TORCH'] = '1'
import torch
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import time
import pandas as pd

from doctr.io import DocumentFile
from doctr.models import ocr_predictor



place = "london"
df  = pd.read_csv(f"../mappilary/train_val/{place}/database/postprocessed.csv")
keys = list(df["key"])

predictor = ocr_predictor(pretrained=True)


def process_image(key):
    doc = DocumentFile.from_images([f"../mappilary/train_val/{place}/database/images/{key}.jpg"])

    result = predictor(doc)
    json_output = result.export()
    json_output = json_output["pages"][0]
    json_output["key"] = key
    return json_output





print(len(keys))


start_time = time.time()
count = 10000
for key in tqdm(keys[3291:count]):
    result = process_image(key)
    with open("textPipleline.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time / count, "seconds")