from PIL import Image
import torch
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# objectProcessor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
# objectModel = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

objectProcessor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
objectModel = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

key_file = "../ephemeral/sign/mtsd_v2_fully_annotated/splits/train.txt"

with open(key_file, "r") as f:
    keys = f.readlines()
keys = [key.strip() for key in keys]


SMOOTH = 1e-6

def iou_numpy(outputs: np.array, labels: np.array):
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    return thresholded.mean()  




def generate_mask_from_json(json_data):
    mask = np.zeros((json_data["height"], json_data["width"]), dtype=np.uint8)  # Initialize empty mask

    for obj in json_data['objects']:
        if obj['label'] == 'other-sign' or True:  # Filter for the desired object label
            xmin, ymin, xmax, ymax = obj['bbox'].values()

            # Convert float coordinates to integer pixel coordinates 
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            # Draw the bounding box on the mask
            mask[ymin:ymax, xmin:xmax] = 1 

    return mask[:, :, None]

def generate_mask_from_boxes(boxes, height, width):

    mask = np.zeros((height,width), dtype=np.uint8)  # Initialize empty mask

    for obj in boxes:
        xmin, ymin, xmax, ymax = obj

        # Convert float coordinates to integer pixel coordinates 
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # Draw the bounding box on the mask
        mask[ymin:ymax, xmin:xmax] = 1 

    return mask[:, :, None]

def run_object_detection(image, text):
    # image = crop_image(image, 0, 0, 1600, 1205)
    # Preprocess the image
    inputs = objectProcessor(text=[text],images=image, return_tensors="pt")

    # Perform inference
    outputs =  objectModel(**inputs)
    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # print(target_sizes)
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = objectProcessor.post_process_object_detection(outputs=outputs, threshold=0.1,  target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    return boxes


def process_image(key):
    json_file = f"../ephemeral/sign/mtsd_v2_fully_annotated/annotations/{key}.json"
    with open(json_file) as f:
        data = json.load(f)
    image_file = '../ephemeral/sign/images/' + key + '.jpg'
    # image_file = f"../ephemeral/sign/images/{key}.jpg"
    boxes = run_object_detection(Image.open(image_file), "a traffic sign")
    return iou_numpy(generate_mask_from_json(data), generate_mask_from_boxes(boxes, data["height"], data["width"]))


def debug_image(key):
    json_file = f"../ephemeral/sign/mtsd_v2_fully_annotated/annotations/{key}.json"
    with open(json_file) as f:
        data = json.load(f)
    image_file = '../ephemeral/sign/images/' + key + '.jpg'
    # image_file = f"../ephemeral/sign/images/{key}.jpg"
    boxes = run_object_detection(Image.open(image_file), "a traffic sign")
    plt.imshow(Image.open(image_file))
    plt.imshow(generate_mask_from_boxes(boxes, data["height"], data["width"]), cmap='Blues', alpha=0.5)  # Use a grayscale colormap
    plt.imshow(generate_mask_from_json(data), cmap='grey', alpha=0.5)  # Use a grayscale colormap
    plt.title('Mask')
    plt.show()

print(len(keys))
with open("output.txt", "a") as f:
    for key in tqdm(keys):
        result = process_image(key)
        f.write(f"{key}: {result}\n")
