from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load Iris dataset and train Logistic Regression model
iris = load_iris()
X = iris.data[:, :2]  # Using only sepal_length and sepal_width
y = iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sepal_length = data.get('sepal_length')
    sepal_width = data.get('sepal_width')
    
    if sepal_length is None or sepal_width is None:
        return jsonify({'error': 'sepal_length and sepal_width are required'}), 400

    features = np.array([[sepal_length, sepal_width]])
    prediction = model.predict(features)[0]
    class_name = iris.target_names[prediction]

    return jsonify({'predicted_class': class_name})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
