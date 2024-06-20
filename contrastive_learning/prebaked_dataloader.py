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
TRACK_CACHE = "/rds/general/user/ao921/ephemeral/cache"
def iter_tracks(dataset_path):
    # Get a sorted list of all JSON files in the folder
    # json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        return data



def generate_pairs(tracks,cache,transformer, resolution=600 ):
    # i = 0
    for track in tracks:
        # i+=1
        # if i > 5000:
        #     break
        track_path = track["filename"]
        urls = track["urls"]
        url = random.choice(urls)   
        map = cache.download(url)
        
        transforms = v2.Compose([

            v2.Resize(size=(resolution, resolution), antialias=True),  # Or Resize(antialias=True)
            # ...
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_track = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.RGB(),  # Ensure RGB
            v2.RandomInvert(1),
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            transforms
        ])

        transforms_map = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.RGB(),  # Ensure RGB
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            # ...
            transforms
        ])
        Xi = transform_track(Image.open(f"{TRACK_CACHE}/{track_path}.png").convert('RGB'))
        Xj = transforms_map(map.convert('RGB'))
        yield Xi, Xj

        # Xj = np.array(map.resize((resolution,resolution)))
        # Xj = np.repeat(Xj[:, :, np.newaxis], 3, axis=2)
        # yield  np.swapaxes(Xi,0,2 ),np.swapaxes(Xj, 0, 2)


class ImageMapDataset(torch.utils.data.IterableDataset):
     def __init__(self, cache_dir:str = "./cache", track_dir:str = "./dataset.json", resolution:int=600):
         super(ImageMapDataset).__init__()
         self.cache_dir = cache_dir
         self.cache = ImageCache(cache_dir)
         self.track_dir = track_dir
         self.resolution = resolution
         self.transformer = Transformer.from_crs("epsg:4326", "epsg:3857")

     def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         tracks_it = iter_tracks(self.track_dir)
         if worker_info is not None:  
             # in a worker process
             # split workload
             start = 0
             end = len(tracks_it)
             per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, end)
             tracks_it = tracks_it[iter_start:iter_end]
         return generate_pairs(tracks_it, self.cache, self.transformer, self.resolution)


class ImageMapDataModule(pl.LightningDataModule):
    def __init__(self, cache_dir: str = "./cache", batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.im_train= ImageMapDataset(track_dir="./dataset.json", cache_dir=self.cache_dir)
            self.im_val = ImageMapDataset(track_dir="./human.json", cache_dir=self.cache_dir)
        if stage == "test":
            self.im_test = ImageMapDataset(track_dir="./human.json", cache_dir=self.cache_dir)
        if stage == "predict":
            self.im_predict = ImageMapDataset(track_dir="./dataset.json", cache_dir=self.cache_dir)
        

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
