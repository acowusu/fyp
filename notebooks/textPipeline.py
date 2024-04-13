import pandas as pd
from pathlib import Path
seed = 3
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
from geopy.geocoders import Nominatim
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import argparse  # Import argparse for command-line arguments

# Construct Argument Parser and add arguments
parser = argparse.ArgumentParser(description="Process image for sign recognition and location lookup")
parser.add_argument("image_path", help="The path to the image file.")
parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed progress messages")
parser.add_argument("-o", "--output_file", help="Path to an output file to write location data")
args = parser.parse_args()

# img = Image.open("sampleImages/sign.jpeg")
img = Image.open(args.image_path)  
         

import easyocr
reader = easyocr.Reader(['en']) 


def crop_image(image, x, y, w, h):
    box = (x, y, x + w, y + h)
    cropped_image = image.crop(box)
    return cropped_image


# load image from the IAM database (actually this model is meant to be used on printed text)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')


if args.verbose:
    print(f"TRocr loaded ")
    
    
seg_image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-panoptic"
)


if args.verbose:
    print(f"mask2former loaded ")
    
    
seg_inputs = seg_image_processor(img, return_tensors="pt")

if args.verbose:
    print(f"mask2former initilaized ")
with torch.no_grad():
    seg_outputs = seg_model(**seg_inputs)

if args.verbose:
    print(f"mask2former inference complete ")
class_queries_logits = seg_outputs.class_queries_logits
masks_queries_logits = seg_outputs.masks_queries_logits

def run_text_extraction(x,y,w,h):
    textSeg = crop_image(img,x,y,w,h)
    pixel_values = processor(images=textSeg, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if args.verbose:
        print(f"trocr inference complete {generated_text}")
    return generated_text


pred_panoptic_map = seg_image_processor.post_process_panoptic_segmentation(
    seg_outputs, target_sizes=[img.size[::-1]]
)[0]

for segment in pred_panoptic_map["segments_info"]:
    if seg_model.config.id2label[segment["label_id"]] == "traffic sign":
            segment_id = segment['id']
            geolocator = Nominatim(user_agent="Imperial College London")
            mask = (pred_panoptic_map['segmentation'].numpy() == segment_id)
            visual_mask = (mask * 255).astype(np.uint8)
            # visual_mask = Image.fromarray(visual_mask)
            # visual_mask
            contours, hierarchy = cv2.findContours(visual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)  
                text = run_text_extraction(x, y, w, h)
                location = geolocator.geocode(text)
                if location is not None:
                    output_string = f"{location.address}\n({location.latitude}, {location.longitude})\n"

                    if args.output_file:  
                        with open(args.output_file, "a") as f:  # 'a' for append mode
                            f.write(output_string)
                    else:
                        print(output_string)

                    if args.verbose:
                        print(f"Location extracted: {output_string}")
                