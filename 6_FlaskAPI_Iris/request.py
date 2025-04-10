import requests
response = requests.post('http://localhost:5000/predict', 
                        json={'sepal_length': 5.1, 'sepal_width': 3.5})
print(response.json())