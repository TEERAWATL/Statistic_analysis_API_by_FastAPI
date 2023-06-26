#Chi-squred test (nonparametic test)
import requests
import json

#load data into payload as json format
payload = json.dumps({
  "v": [
    "male",
    "female",
    "male",
    "male",
    "female",
    "male"
  ],
  "b": [
    "a",
    "b",
    "a",
    "a",
    "b",
    "b"
  ]
})

# Define the headers for the request, including the token
headers = {
    "Authorization": "Bearer teerawat12345",
    "Content-Type": "application/json"
}

# URL of the API endpoint
url = "https://stats-api-okdirtqcca-as.a.run.app/chi_squared"

# Send POST request with data and headers
try:
    response = requests.post(url, data=payload, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Print the result
    print(response.json())

#print error message if request failed
except requests.exceptions.HTTPError as err:
    print(f"Request failed with status code {response.status_code}")
    print(err)
