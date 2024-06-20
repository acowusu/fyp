import json
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
import os
START_PAGE = 0
# check if we are resuming by counting the number of files in "tracks/human"
if len(os.listdir("tracks/synthetic")) > 0:
    START_PAGE = len(os.listdir("tracks/synthetic"))
    print(f"Resuming from page {START_PAGE}")


# 971 is the last page
END_PAGE = 971


def parse_roads(elements):
    adjacency_list = {}
    node_to_coords = {}

    for road in elements:
        nodes = road['nodes']
        geometry = road['geometry']
        
        # Fill node_to_coords dictionary
        for node, coord in zip(nodes, geometry):
            node_to_coords[node] = (coord['lat'], coord['lon'])
        
        # Fill adjacency list
        for i in range(len(nodes) - 1):
            if nodes[i] not in adjacency_list:
                adjacency_list[nodes[i]] = []
            if nodes[i + 1] not in adjacency_list:
                adjacency_list[nodes[i + 1]] = []
            adjacency_list[nodes[i]].append(nodes[i + 1])
            adjacency_list[nodes[i + 1]].append(nodes[i])
    
    return adjacency_list, node_to_coords


def generate_random_walk(adjacency_list, node_to_coords, start_node, walk_length):
    current_node = start_node
    walk_nodes = [current_node]
    walk_coords = [{'lat':node_to_coords[current_node][0], 'lon':node_to_coords[current_node][1]} ]

    for _ in range(walk_length - 1):
        if current_node not in adjacency_list or len(adjacency_list[current_node]) == 0:
            break  # No more nodes to walk to
        
        next_node = random.choice(adjacency_list[current_node])
        walk_nodes.append(next_node)
        walk_coords.append( {'lat':node_to_coords[next_node][0], 'lon':node_to_coords[next_node][1]})
        current_node = next_node
    
    return  walk_coords

def generate_batch(id, count, adjacency_list, node_to_coords):
    # array([51.4, -0.3105547])
    # array([51.6,  0.0837953])

    tracks = []
    for i in range(count):
        start_node = random.choice(list(adjacency_list.keys()))
        points = generate_random_walk(adjacency_list, node_to_coords, start_node, random.randint(100, 1000))
        track_info = {
                'name': f"track_{id}_{i}", 
                'desc': "SYNTETIC TRACK",
                'points': points
            }
        tracks.append(track_info)

    return tracks

def download_tracks(adjacency_list, node_to_coords):
    for i in tqdm(range(START_PAGE, END_PAGE)):
        tracks = generate_batch(i, 100, adjacency_list, node_to_coords)
        with open(f"tracks/synthetic/tracks_{i}.json", "w") as f:
            json.dump(tracks, f)

if __name__ == "__main__":
    with open("roads.geojson") as f:
        data = json.loads(f.read())
    adjacency_list, node_to_coords = parse_roads(data['elements'])

    download_tracks(adjacency_list, node_to_coords)