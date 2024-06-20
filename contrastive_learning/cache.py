import os
import hashlib
import requests
from PIL import Image
from io import BytesIO

class ImageCache:
    def __init__(self, cache_dir = 'cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_path(self, url):
        # Use a hash of the URL as the filename to ensure it's unique
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.png")

    def download(self, url):
        cache_path = self._get_cache_path(url)
        
        if os.path.exists(cache_path):
            # If the image is already cached, load and return it
            return Image.open(cache_path)
        else:
            # Download the image
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            
            # Save the image to the cache
            image.save(cache_path)
            return image
