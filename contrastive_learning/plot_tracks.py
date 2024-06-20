import math
# import torch
from utils import lat_lon_to_tile, get_tile_urls
import os
import json
# from cache import ImageCache
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# from torch.utils.data import DataLoader
# import pytorch_lightning as pl
from pyproj import Transformer
# from torchvision.transforms import v2
random.seed(1)
import hashlib
from tqdm import tqdm
from pathlib import Path

def iter_tracks(folder_path, json_files):
    # Get a sorted list of all JSON files in the folder
    # json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for track in data:
                yield track



def generate_plot(track,cache_dir,transformer, resolution=600 ):
        track_points = track['points']
        x_coords = []
        y_coords = []
        for point in track_points:
            lon = point['lon']
            lat = point['lat']
            x, y = transformer.transform(lat, lon)
            x_coords.append(x)
            y_coords.append(y)

        # Plot using Matplotlib
        fig = plt.figure(figsize=(resolution/64, resolution/64), dpi=64)
        ax = fig.add_subplot()
        ax.plot( x_coords,y_coords,linewidth=1, color = "black")
        ax.axis('off')
    
        fig.gca().set_aspect('equal', adjustable='box')
        ax.grid(True)
        
        lat_min = min([point['lat'] for point in track_points])
        lon_min = min([point['lon'] for point in track_points])
        lat_max = max([point['lat'] for point in track_points])
        lon_max = max([point['lon'] for point in track_points])
        zoom = 14

        urls = get_tile_urls(lat_min, lon_min, lat_max, lon_max, zoom)
        nonce  = "-".join(map(str, x_coords)) + "/"  + "-".join(map(str, y_coords))
        filename = hashlib.md5(nonce.encode('utf-8')).hexdigest()
#         print(filename)
        fig.savefig(cache_dir / f"{filename}.png", dpi=fig.dpi)
        plt.close(fig)
        
        return {"filename":filename,"urls": urls}




if __name__ == "__main__":
    cache_dir = Path("/rds/general/user/ao921/ephemeral/cache")
    track_dir  = "./tracks/human"
    json_files = sorted([f for f in os.listdir(track_dir) if f.endswith('.json')])
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    results = []
    for track in tqdm(iter_tracks(track_dir, json_files)):
        results.append(generate_plot(track,cache_dir, transformer))
                      
    with open("human.json", "w") as f:
        f.write(json.dumps(results, indent=2))
        
        