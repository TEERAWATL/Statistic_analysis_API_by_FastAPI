# Request for Descriptive Statistic (numerical)
import requests
import base64
from io import BytesIO
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# load data into payload as json format
payload = json.dumps({
  "dataaaaaaa": [
    1,
    2,
    3,
    4,
    5
  ]
})

# API endpoint URL
url = "https://stats-api-okdirtqcca-as.a.run.app/numerical_descriptive"

# Define the headers, including the token
headers = {
    "Authorization": "Bearer teerawat12345"
}

# Send POST request with data and headers
try:
    response = requests.post(url, data=payload, headers=headers)
    response.raise_for_status()  # Raise an exception for non-2xx status codes

    # Extract histogram image and statistics from the response
    result = response.json()
    histogram_image_base64 = result['histogram']
    statistics = result['statistics']

    # Decode and display the histogram image
    image_data = base64.b64decode(histogram_image_base64)
    image = Image.open(BytesIO(image_data))

    # Display the image
    plt.figure(figsize=(8,6))
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()

    # Display the descriptive statistics
    print("Descriptive Statistics:")
    statistics_df = pd.DataFrame(statistics)
    print(statistics_df)

# Catch exceptions encountered during the request, such as a 500 or connection error
except requests.exceptions.HTTPError as err:
    print(f"Request failed with status code {response.status_code}")
    print(err)
