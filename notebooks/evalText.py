import pandas as pd
from pathlib import Path
base_dir =  Path("/rds/general/user/ao921/home/eval_datasets")

from typing import Any, Iterator
from PIL import Image

import os
os.environ['USE_TORCH'] = '1'
import numpy as np
import json
from tqdm import tqdm
import time

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

DATASET_NAME = "yfcc4k"
class ImageDataset():
  def __init__(self, image_dir: Path, label_file:Path) -> None:
    self.image_dir = image_dir
    self.labels = pd.read_csv(label_file)
#     self.image_paths = [path for path in image_dir.iterdir() if path.is_file()]

  def __len__(self) -> int:
    return len(self.labels)

  def __iter__(self) -> Iterator[Image.Image]:
    for row in range(len(self.labels)):
      yield self.load_image(self.labels.iloc[row])

  def load_image(self, row ) -> Image.Image:
    return {
#         "image" : Image.open(self.image_dir / row["IMG_ID"]),
        "id": row["IMG_ID"],
        "path" : self.image_dir / row["IMG_ID"],
        "lat": row["LAT"],
        "lon": row["LON"]
    }

  def __getitem__(self, item):
    return self.load_image(self.labels.iloc[item]) 

# predictor = ocr_predictor(pretrained=True)
image_dataset = ImageDataset(base_dir/ DATASET_NAME, base_dir / f"{DATASET_NAME}.csv")


predictor = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)

def only_punctuation(text):
  valid_chars = set(" -:,.!?()[]{}'\"/\\&%$@#*+=")
  return all(char in valid_chars for char in text)

def getPhotoLines(doc):
    items = []
    for block in doc["blocks"]:  
        for line in block["lines"]:
            text = ""
            for word in line["words"]:
                text =  text +" "+ word["value"]
            if only_punctuation(text) or len(text )< 3:
                pass
            else:
                items.append(text)
    return items


def process_image(image):
    doc = DocumentFile.from_images([image["path"]])

    result = predictor(doc).export()["pages"][0]
    lines = getPhotoLines(result)
    output  = {}
    output["full"] = result
    output["key"] = image["id"]
    output["lines"] = lines
    output["lat"] = image["lat"]
    output["lon"] = image["lon"]
    return output

# print(json.dumps(process_image(image_dataset[1])))

print("STATRING")




start_time = time.time()
batch_size = 100
data_to_write = []

for key in tqdm(image_dataset):
  try:
      result = process_image(key)
      data_to_write.append(json.dumps(result) + "\n")
  except FileNotFoundError as e:
      print(f"Error: File not found - {e}")

  # Write every 100 iterations or at the end of the loop
  if len(data_to_write) % batch_size == 0:
    with open(f"{DATASET_NAME}.jsonl", "a") as f:
      f.writelines(data_to_write)
    data_to_write = []  # Clear the batch list

with open(f"{DATASET_NAME}.jsonl", "a") as f:
    f.writelines(data_to_write)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time / len(image_dataset), "seconds")