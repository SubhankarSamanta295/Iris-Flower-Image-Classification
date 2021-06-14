import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Sepal Length':4, 'Sepal Width':3, 'Petal Length':1, 'Petal Width':1})

print(r.json())