# prevalence calculation
import requests
import json

#Set the url for the API
url = "https://stats-api-okdirtqcca-as.a.run.app/prevalence"

#load data into payload as json format
payload = json.dumps({
  "df": {
    "สภาวะเปราะบาง": [
      "มีสภาวะเปราะบาง",
      "ไม่มีสภาวะเปราะบาง",
      "ไม่มีสภาวะเปราะบาง",
      "มีสภาวะเปราะบาง",
      "มีสภาวะเปราะบาง",
      "มีสภาวะเปราะบาง",
      "ไม่มีสภาวะเปราะบาง",
      "ไม่มีสภาวะเปราะบาง",
      "มีสภาวะเปราะบาง"
    ]
  },
  "column_name": "สภาวะเปราะบาง"
})

# Set the headers and private key
headers = {
  'Authorization': 'Bearer teerawat12345',
  'Content-Type': 'application/json'
}

# Send the request
response = requests.request("POST", url, headers=headers, data=payload)

# Check if the request was successful
if response.status_code == 200:
    # If successful, print the result
    print(response.json())
else:
    # If unsuccessful, print the status code and message
    print("Error:", response.status_code)
    print("Message:", response.text)
