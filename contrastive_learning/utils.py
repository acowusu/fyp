import math

MAPBOX_USER = "ao921"
MAPBOX_STYLE = "clwi9a3wl00qf01qs5yxxaqau"
MAPBOX_ACCESS_TOKEN = "pk.eyJ1IjoiYW85MjEiLCJhIjoiY2xtY3B3bXlmMGZzcDNjbGt4NnJkcThlcCJ9.hRmRQxruud7nPpLZ-vsQRw"

def lat_lon_to_tile(lat, lon, zoom):
    """
    Converts latitude and longitude to tile coordinates at a specified zoom level.
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x_tile, y_tile


def get_tile_urls(lat_min, lon_min, lat_max, lon_max, zoom):
    """
    Returns the tile URLs for the given latitude-longitude bounding box and zoom level.
    """
    x_1, y_1 = lat_lon_to_tile(lat_max, lon_min, zoom)
    x_2, y_2 = lat_lon_to_tile(lat_min, lon_max, zoom)
    
    x_min = min(x_1, x_2)
    y_min = min(y_1, y_2)
    x_max = max(x_1, x_2)
    y_max = max(y_1, y_2)    
    tile_urls = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_url = f"https://api.mapbox.com/styles/v1/{MAPBOX_USER}/{MAPBOX_STYLE}/tiles/{zoom}/{x}/{y}?access_token={MAPBOX_ACCESS_TOKEN}"
            tile_urls.append(tile_url)
    
    return tile_urls

if __name__ == "main":
    print("loaded utils")