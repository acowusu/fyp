import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import json
from geopy import distance
from geopy.geocoders import Nominatim, Photon, Pelias
from geopy.extra.rate_limiter import RateLimiter

nominatim = Nominatim(user_agent="Imperial College London - RCS", )
photon = Photon(user_agent="Imperial College London - RCS",)
pelias = Pelias(api_key="ge-f3815bf1ea32a3de", domain="https://api.geocode.earth/v1/search")


im2gps = pd.read_json("~/notebooks/im2gps_small.jsonl", lines=True)
im2gps = im2gps.set_index('key')
# im2gps3k = pd.read_json("~/notebooks/im2gps3ktest.jsonl", lines=True)
# im2gps3k = im2gps3k.set_index('key')
# yfcc4k = pd.read_json("~/notebooks/yfcc4k.jsonl", lines=True)
# yfcc4k = yfcc4k.set_index('key')

im2gps.drop(columns=['full'], inplace=True)
# im2gps3k.drop(columns=['full'], inplace=True)
# yfcc4k.drop(columns=['full'], inplace=True)


im2gps['num_lines'] = im2gps['lines'].apply(len)
# im2gps3k['num_lines'] = im2gps3k['lines'].apply(len)
# yfcc4k['num_lines'] = yfcc4k['lines'].apply(len)

im2gps_filtered   = im2gps[im2gps['num_lines'] > 0]
# im2gps3k_filtered = im2gps3k[im2gps3k['num_lines'] > 0]
# yfcc4k_filtered   = yfcc4k[yfcc4k['num_lines'] > 0]

# yfcc4k_locations = pd.read_json("~/notebooks/yfcc4k_locations.jsonl", lines=True)
# yfcc4k_locations = yfcc4k_locations.set_index('key')
# yfcc4k_filtered =  yfcc4k_filtered.loc[~yfcc4k_filtered.index.isin(yfcc4k_locations.index)]

# im2gps3k_locations = pd.read_json("~/notebooks/im2gps3k_locations.jsonl", lines=True)
# im2gps3k_locations = im2gps3k_locations.set_index('key')
# im2gps3k_filtered =  im2gps3k_filtered.loc[~im2gps3k_filtered.index.isin(im2gps3k_locations.index)]

# im2gps_locations = pd.read_json("~/notebooks/im2gps_locations.jsonl", lines=True)
# im2gps_locations = im2gps_locations.set_index('key')
# im2gps_filtered =  im2gps_filtered.loc[~im2gps_filtered.index.isin(im2gps_locations.index)]

from pathlib import Path

from typing import Any, Iterator
from PIL import Image

import numpy as np
from tqdm import tqdm
import time

base_dir =  Path("/rds/general/user/ao921/home/notebooks")

photon_geocode = RateLimiter(photon.geocode, min_delay_seconds=1)
nominatim_geocode = RateLimiter(nominatim.geocode, min_delay_seconds=1)
pelias_geocode =  RateLimiter(pelias.geocode, min_delay_seconds=1)

def get_candidates(lines, providor):
    results = []
    for line in lines:
        time.sleep(1)
        items = providor(line, exactly_one=False)
        if items:
            results.extend(items)
    for i in range(len(lines)-1):
        line = lines[i] + ' ' + lines[i+1]
        items = providor(line, exactly_one=False)
        time.sleep(1)

        if items:
            results.extend(items)
    return results


def process_(row):
    key, data = row
#     key, lines, lat , lon, numlines = row
    print(key)
    locations_nominatim = [{"lat" : l.latitude, "lon": l.longitude} for l in get_candidates(data['lines'], nominatim_geocode)]
    locations_photon = [{"lat" : l.latitude, "lon": l.longitude} for l in get_candidates(data['lines'], photon_geocode)]

    return {"key": key, 
            "locations_nominatim": locations_nominatim, 
            "locations_photon": locations_photon,
            "lat":data['lat'], 
            "lon": data['lon']}

# print(json.dumps(process_image(image_dataset[1])))

print("STATRING")


DATASET_NAME = "im2gps"
df = im2gps_filtered

start_time = time.time()
batch_size = 1
data_to_write = []

for key in tqdm(df.iterrows()):
  try:
      result = process_(key)
      data_to_write.append(json.dumps(result) + "\n")
  except FileNotFoundError as e:
      print(f"Error: File not found - {e}")

  # Write every 100 iterations or at the end of the loop
  if len(data_to_write) % batch_size == 0:
    with open(f"{DATASET_NAME}_locations.jsonl", "a") as f:
      f.writelines(data_to_write)
    data_to_write = []  # Clear the batch list

with open(f"{DATASET_NAME}.jsonl", "a") as f:
    f.writelines(data_to_write)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time / len(df), "seconds")