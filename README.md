
### **Q1. Frequency Counter**
**Description:**  
Counts the frequency of each number in a list.

### **Q2. Min-Max Normalization**
**Description:**  
Normalizes a list of numbers between 0 and 1 using min-max scaling.


### **Q3. Salary Data Cleaning (Pandas)**
**Description:**  
- Generates a sample `salaries.csv` file  
- Fills missing salary values with the column mean  
- Calculates the average salary after cleaning

### **Q4. Iris Classification (Scikit-learn)**
**Description:**  
- Loads the Iris dataset  
- Trains a logistic regression model  
- Prints the accuracy and classification report

### **Q5. Prompt Engineering (LLM)**
**Description:**  
Example prompt to summarize a long email into bullet points for a large language model.

### **Q6. Flask API – Iris Prediction**
![image](https://github.com/user-attachments/assets/f0e1b339-c7d8-43fe-a46a-d2d217c55904)

**Description:**  
A simple Flask API with a `/predict` endpoint that returns the predicted Iris class based on `sepal_length` and `sepal_width`.

**To run:**

1. Make sure required packages are installed:
```bash
pip install flask scikit-learn numpy
```

2. Run the app:
```bash
python app.py
```

3. Make a POST request to:
```
http://127.0.0.1:5000/predict
```
4. Run the request.py file in the folder.
**Sample JSON Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5
}
```

**Expected Response:**
```json
{
  "predicted_class": "setosa"
}
```

---
