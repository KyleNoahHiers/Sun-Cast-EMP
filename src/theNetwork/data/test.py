import requests

url = 'https://us-central1-spartan-alcove-415521.cloudfunctions.net/load_predict'
data = {'month': 3, 'day': 31}
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status() # Raise an exception for 4xx or 5xx status codes
    json_response = response.json()
    print(json_response)
except requests.exceptions.RequestException as e:
    print("Error:", e)
except ValueError as ve:
    print("Error decoding JSON response:", ve)