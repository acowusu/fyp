import json
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
START_PAGE = 0
# check if we are resuming by counting the number of files in "tracks/human"
if len(os.listdir("tracks/human")) > 0:
    START_PAGE =   len(os.listdir("tracks/human")) - 1
    print(f"Resuming from page {START_PAGE}")


# 971 is the last page
END_PAGE = 971

def fetch_osm_tracks(page=0):
    # array([51.4, -0.3105547])
    # array([51.6,  0.0837953])


    api_url = f"https://api.openstreetmap.org/api/0.6/trackpoints?bbox=-0.20,51.4,0.05,51.65&page={page}"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")

    gpx_data = response.content
    root = ET.fromstring(gpx_data)
    
    tracks = []

    for trk in root.findall('{http://www.topografix.com/GPX/1/0}trk'):
        track_name = trk.find('{http://www.topografix.com/GPX/1/0}name')
        if track_name is None:
            track_name = "unknown"
        else:
            track_name = track_name.text
        track_desc = trk.find('{http://www.topografix.com/GPX/1/0}desc')
        if track_desc is None:
            track_desc = "unknown"
        else:
            track_desc = track_desc.text
        track_url = trk.find('{http://www.topografix.com/GPX/1/0}url')
        if track_url is not None:
            track_url = track_url.text
        else:
            track_url = "unknown"
        points = []

        for trkseg in trk.findall('{http://www.topografix.com/GPX/1/0}trkseg'):
            for trkpt in trkseg.findall('{http://www.topografix.com/GPX/1/0}trkpt'):
                time = trkpt.find('{http://www.topografix.com/GPX/1/0}time')
                if time is not None:
                    lat = float(trkpt.attrib['lat'])
                    lon = float(trkpt.attrib['lon'])
                    points.append({'lat': lat, 'lon': lon})

        if points:
            track_info = {
                'name': track_name,
                'desc': track_desc,
                'url': track_url,
                'points': points
            }
            tracks.append(track_info)

    return tracks

def download_tracks():
    for i in tqdm(range(START_PAGE, END_PAGE)):
        tracks = fetch_osm_tracks(i)
        with open(f"tracks/human/tracks_{i}.json", "w") as f:
            json.dump(tracks, f)

if __name__ == "__main__":
    download_tracks()
    pass