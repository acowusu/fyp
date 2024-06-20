                   import requests
import urllib.parse
import json
# Define the bounding box coordinates
bbox = (51.40456, -0.288734, 51.595628, 0.073471)
bbox_str = ','.join(map(str, bbox))
# Build the Overpass QL query
query = f"""[bbox:{bbox_str}]
[out:json]
[timeout:90];

way["highway"]({bbox_str});
out geom;
"""



# Encode the query
encoded_query = urllib.parse.quote_plus(query.strip())

# Prepare the request
url = "https://overpass-api.de/api/interpreter"
headers = {"Content-Type": "application/x-www-form-urlencoded"}
data = f"data={encoded_query}"

# Send the POST request
response = requests.post(url, headers=headers, data=data)

# Check for successful response
if response.status_code == 200:
  # Parse the JSON data
  result = response.json()
  # Print the formatted JSON response
  with open("roads.geojson", "w") as f:
        f.write(json.dumps(result, indent=2))
else:
  # Handle error
  print(f"Error: {response.status_code}")
