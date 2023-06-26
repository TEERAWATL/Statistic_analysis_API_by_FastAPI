#Preson Correlation test for 2 variables (numerical)
import requests
import json

# load data into payload as json format
payload = json.dumps({
  "group1": [
    1,
    2,
    1,
    1,
    1,
    5,
    5
  ],
  "group2": [
    5,
    6,
    5,
    5,
    5,
    10,
    2
  ]
})

# Define the headers for the request, including the token
headers = {
    "Authorization": "Bearer teerawat12345",
    "Content-Type": "application/json"
}

# URL of the API endpoint
url = "https://stats-api-okdirtqcca-as.a.run.app/correlation"

# Send POST request with data and headers
try:
    response = requests.post(url, data=payload, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Print the result
    print(response.json())

# Catch exceptions encountered during the request, such as a 500 or connection error
except requests.exceptions.HTTPError as err:
    print(f"Request failed with status code {response.status_code}")
    print(err)
