import math
import torch
from utils import lat_lon_to_tile, get_tile_urls
import os
import json
from cache import ImageCache
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pyproj import Transformer
from torchvision.transforms import v2

def iter_tracks(folder_path, json_files):
    # Get a sorted list of all JSON files in the folder
    # json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for track in data:
                yield track



def generate_pairs(tracks,cache,transformer, resolution=600 ):
    for track in tracks:
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
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.plot( x_coords,y_coords,linewidth=1, color = "black")
        ax.axis('off')
    
        fig.gca().set_aspect('equal', adjustable='box')
        ax.grid(True)
        
        canvas.draw()
        plt.close(fig)


        # Pass off to PIL.
        Xi = 255 - np.asarray(canvas.buffer_rgba())[:, :, :3]

        lat_min = min([point['lat'] for point in track_points])
        lon_min = min([point['lon'] for point in track_points])
        lat_max = max([point['lat'] for point in track_points])
        lon_max = max([point['lon'] for point in track_points])
        zoom = 14

        urls = get_tile_urls(lat_min, lon_min, lat_max, lon_max, zoom)
        url = random.choice(urls)   
        map = cache.download(url)
        
        transforms = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            # v2.RGB(),  # Ensure RGB
            v2.ToDtype(torch.uint8),  # optional, most input are already uint8 at this point
            # ...
            v2.Resize(size=(resolution, resolution), antialias=True),  # Or Resize(antialias=True)
            # ...
            v2.ToDtype(torch.float32),  # Normalize expects float input
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        Xi = transforms(Xi)
        Xj = transforms(map.convert('RGB'))
        yield Xi, Xj

        # Xj = np.array(map.resize((resolution,resolution)))
        # Xj = np.repeat(Xj[:, :, np.newaxis], 3, axis=2)
        # yield  np.swapaxes(Xi,0,2 ),np.swapaxes(Xj, 0, 2)


class ImageMapDataset(torch.utils.data.IterableDataset):
     def __init__(self, cache_dir:str = "./cache", track_dir:str = "./tracks/synthetic", resolution:int=600):
         super(ImageMapDataset).__init__()
         self.cache_dir = cache_dir
         self.cache = ImageCache(cache_dir)
         self.track_dir = track_dir
         self.resolution = resolution
         self.transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
         self.json_files = sorted([f for f in os.listdir(track_dir) if f.endswith('.json')])

     def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None:  # single-process data loading, return the full iterator
             tracks_it = iter_tracks(self.track_dir, self.json_files)
             
         else:  # in a worker process
             # split workload
             start = 0
             end = len(self.json_files)
             per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, end)
             tracks_it = iter_tracks(self.track_dir, self.json_files[iter_start:iter_end])
         return generate_pairs(tracks_it, self.cache, self.transformer, self.resolution)


class ImageMapDataModule(pl.LightningDataModule):
    def __init__(self, cache_dir: str = "./cache", batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.im_train= ImageMapDataset(track_dir="./tracks/synthetic", cache_dir=self.cache_dir)
            self.im_val = ImageMapDataset(track_dir="./tracks/human", cache_dir=self.cache_dir)
        if stage == "test":
            self.im_test = ImageMapDataset(track_dir="./tracks/human", cache_dir=self.cache_dir)
        if stage == "predict":
            self.im_predict = ImageMapDataset(track_dir="./tracks/human", cache_dir=self.cache_dir)
        

    def train_dataloader(self):
        return DataLoader(self.im_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.im_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.im_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.im_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

if __name__ == "__main__":
    # Test the ImageMapDataset
    dataset = ImageMapDataset()
    dataloader = DataLoader(dataset, batch_size=2)
    for X, Y in dataloader:
        print(X.shape, Y.shape)
        break
    # Test the ImageMapDataModule
    data_module = ImageMapDataModule(batch_size=2)
    data_module.setup("fit")
    dataloader = data_module.train_dataloader()
    for X, Y in dataloader:
        print(X.shape, Y.shape)
        break
