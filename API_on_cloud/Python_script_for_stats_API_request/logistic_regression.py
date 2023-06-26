#Logistic regression test
import requests
import json

# load data into payload as json format
payload = json.dumps({
  "input_data": {
    "column11": [
      1,
      0,
      1,
      0,
      1,
      1
    ],
    "column22": [
      5,
      10,
      11,
      12,
      15,
      20
    ]
  },
  "dependent_variable": "column11",
  "independent_variables": [
    "column22"
  ]
})

# Define the headers for the request, including the token
headers = {
    "Authorization": "Bearer teerawat12345",
    "Content-Type": "application/json"
}

# URL of the API endpoint
url = "https://stats-api-okdirtqcca-as.a.run.app/logistic_regression"

# Send POST request with data and headers
response = requests.post(url, headers=headers, data=payload)

# Check if the request was successful
if response.status_code == 200:
    # If successful, print the result
    print(response.json())
else:
    # If unsuccessful, print the status code and message
    print("Error:", response.status_code)
    print("Message:", response.text)
