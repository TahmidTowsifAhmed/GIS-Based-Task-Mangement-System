import pandas as pd
import requests
import folium
import numpy as np
from folium.plugins import HeatMap

# Step 1: Read the CSV file
df = pd.read_csv('all_columns_Hilleroed_processed.CSV', encoding='ISO-8859-1')
df = df.replace({np.nan: None})  # Replace NaN values with None
latitudes = df['latitude']
longitudes = df['longitude']
task_numbers = df['task_number']
descriptions = df['description']

# Step 2: Perform reverse geocoding with Mapbox Geocoding API
base_url = 'https://api.mapbox.com/geocoding/v5/mapbox.places'

access_token = 'pk.eyJ1IjoidGFobWlkdG93c2lmYWhtZWQiLCJhIjoiY2xoeHN3ZXlkMHd2azNrcGN3aHlqczFuMiJ9.OB62Llwwjv61_5rmbyz5ig'

addresses = []
for lat, lon in zip(latitudes, longitudes):
    if pd.notnull(lat) and pd.notnull(lon):  # Skip rows with missing lat or lon
        url = f'{base_url}/{lon},{lat}.json'
        params = {'access_token': access_token, 'limit': 1}
        response = requests.get(url, params=params)
        data = response.json()
        if 'features' in data and len(data['features']) > 0:
            address = data['features'][0]['place_name']
            addresses.append(address)
        else:
            addresses.append('Not Found')
    else:
        addresses.append('Not Found')

# Step 3: Generate the web map and zoom to Denmark
denmark_coordinates = (55.7761, 12.5683)  # Latitude, Longitude of Denmark
zoom_level = 10

# Create the map
map = folium.Map(location=denmark_coordinates, zoom_start=zoom_level)

# Add the title on the map using custom HTML
title_html = '''
             <h3 align="center" style="font-size:16px"><b>Task Locations</b></h3>
             '''
map.get_root().html.add_child(folium.Element(title_html))

# Create a FeatureGroup layer for the markers
marker_layer = folium.FeatureGroup(name='Markers').add_to(map)

# Add markers as small dots
for lat, lon, task_num, address, description in zip(latitudes, longitudes, task_numbers, addresses, descriptions):
    if pd.notnull(lat) and pd.notnull(lon):  # Skip rows with missing lat or lon
        popup_text = f'Task Number: {task_num}<br>Address: {address}<br>Task description: {description}'
        folium.CircleMarker([lat, lon], radius=2, color='blue', fill=True, fill_color='blue', fill_opacity=1,
                            popup=popup_text).add_to(marker_layer)

# Create HeatMap layer based on the points
heatmap_data = [[lat, lon] for lat, lon in zip(latitudes, longitudes) if pd.notnull(lat) and pd.notnull(lon)]
HeatMap(heatmap_data, name='heatmap').add_to(map)

# Add layer control
folium.LayerControl().add_to(map)

# Save the map
map.save('hillerød.html')

# Read the HTML file
with open('hillerød.html', 'r') as file:
    content = file.read()

print("Web map generated and saved as 'hillerød.html' with the title 'Task Locations' on the map.")














