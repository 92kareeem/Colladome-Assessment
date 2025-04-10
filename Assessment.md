1. Frequency Counter

```python
def frequency_counter(numbers):
    """
    Returns a dictionary with the frequency count of each number in the list.
    """
    freq = {}
    for num in numbers:
        freq[num] = freq.get(num, 0) + 1
    return freq

# Example usage
if __name__ == "__main__":
    sample = [1, 2, 2, 3, 1, 4]
    print("Frequency Count:", frequency_counter(sample))
```

---

2. Min-Max Normalization

```python
def min_max_normalize(numbers):
    """
    Returns a list of numbers normalized between 0 and 1 using min-max scaling.
    """
    min_val = min(numbers)
    max_val = max(numbers)
    if max_val == min_val:
        return [0 for _ in numbers]  # avoid division by zero
    return [(x - min_val) / (max_val - min_val) for x in numbers]

# Example usage
if __name__ == "__main__":
    sample = [10, 20, 30, 40]
    print("Normalized:", min_max_normalize(sample))
```

---

3. Salary Data Cleaning (Pandas)

```python
import pandas as pd

def clean_salary_data(csv_path):
    """
    Loads CSV, fills missing salaries with mean, returns average salary after cleaning.
    """
    df = pd.read_csv(csv_path)
    if 'salary' in df.columns:
        df['salary'].fillna(df['salary'].mean(), inplace=True)
        return df['salary'].mean()
    else:
        raise KeyError("The column 'salary' does not exist in the file.")

# Example usage:
# print("Average Salary:", clean_salary_data('salaries.csv'))
```

---

4. Iris Classification (Scikit-learn)

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_iris_logistic_model():
    """
    Loads the Iris dataset, trains a logistic regression model, and prints the accuracy.
    """
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
    return model

# model = train_iris_logistic_model()
```

---

5. Prompt Engineering (LLM)

```txt
Prompt:

You are a professional assistant. Summarize the following email into clear and concise bullet points. Highlight key updates, action items, and deadlines if mentioned.

Email:
[Insert the full email content here]

Format:
- Bullet point summary
- Use professional and concise language
```

---

6. Flask API – Iris Prediction

```python
from flask import Flask, request, jsonify
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load and train model
iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data[:, :2], iris.target)  # Using sepal_length and sepal_width only

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON with 'sepal_length' and 'sepal_width'.
    Returns predicted iris class name.
    """
    try:
        data = request.get_json()
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        features = np.array([[sepal_length, sepal_width]])
        prediction = model.predict(features)[0]
        return jsonify({'predicted_class': iris.target_names[prediction]})
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run app
# if __name__ == '__main__':
#     app.run(debug=True)
```
