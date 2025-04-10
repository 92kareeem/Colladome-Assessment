from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_iris_logistic_model():
#1. Load the Iris Dataset
    iris = load_iris()
#2. Split the Dataset using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)
#3. Train the logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
#3. Print accuracy score
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    return model
