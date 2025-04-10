from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and train the model (same as Q4)
def train_iris_logistic_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    return model

# Load the model when starting the app
model = train_iris_logistic_model()
iris = load_iris()
target_names = iris.target_names.tolist()  # ['setosa', 'versicolor', 'virginica']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Extract features (using only sepal_length and sepal_width as requested)
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        
        # Note: Since the model was trained on all 4 features, we need to provide
        # placeholder values (0) for the missing petal features
        features = [[sepal_length, sepal_width, 0, 0]]
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = target_names[prediction[0]]
        
        # Return prediction
        return jsonify({
            'predicted_class': predicted_class,
            'sepal_length': sepal_length,
            'sepal_width': sepal_width
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)