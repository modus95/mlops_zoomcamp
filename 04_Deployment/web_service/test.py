import requests

ride = {"PUlocationID":40, "DOlocationID":50}

url = 'http://localhost:9696/predict'

response = requests.post(url, json=ride)
print(response.json())
