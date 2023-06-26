# Request for categorical data description
import requests
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np

# load data into payload as json format
payload = {
  "data": [
    "ชาย",
    "หญิง",
    "ชาย",
    "หญิง",
    "ชาย",
    "ชาย",
    "หญิง"
  ]
}

# Setup API endpoint URL
url = "https://stats-api-okdirtqcca-as.a.run.app/categorical_plot"

# Define the headers, including the token
headers = {
  'Authorization': 'Bearer teerawat12345',
  'Content-Type': 'application/json'
}

# Send POST request with data and headers
try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an exception for non-2xx status codes

    # Extract histogram image and frequency table from the response
    result = response.json()
    histogram_image_base64 = result['bar_chart']
    freq_table_json = result['freq_table']

    # Decode and display the histogram image
    image_data = base64.b64decode(histogram_image_base64)
    image = Image.open(BytesIO(image_data))

    # Display the image
    plt.figure(figsize=(8,6))
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()

    # Extract the list of dictionaries from the frequency table
    freq_table_list = freq_table_json

    # Create an empty list to store the extracted data
    values =[]
    count = []

    # Iterate over each item in the frequency table list
    for item in freq_table_list:
        # Extract the 'Values' and 'count' from each item and append them to the lists 
        values.append(item['Values'])
        count.append(item['count'])
    
    freq_df = pd.DataFrame({'Values': values, 'count': count})

    print(freq_df)

# Catch exceptions encountered during the request, such as a 500 or connection error
except requests.exceptions.HTTPError as err:
    print(f"Request failed with status code {response.status_code}")
    print(err)
