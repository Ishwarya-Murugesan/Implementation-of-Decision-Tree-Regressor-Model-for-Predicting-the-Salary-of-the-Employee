# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and Load Dataset
2. Prepare and Split the Data
3. Train the Decision Tree Regressor Model
4. Predict and Evaluate the Model

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ISHWARYA M
RegisterNumber:  212225230107
*/
# Decision Tree Regressor Model for Predicting Employee Salary

# Import required libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Salary.csv")

# Display dataset
print("Dataset Preview:")
print(data.head())

# Independent variable (Level)
X = data[['Level']]

# Dependent variable (Salary)
y = data['Salary']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict test values
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict salary for a new employee level
new_level = pd.DataFrame([[6]], columns=['Level'])
predicted_salary = model.predict(new_level)

print("\nPredicted Salary for Level 6:")
print(predicted_salary[0])

# Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Decision Tree Prediction')

plt.title("Decision Tree Regressor - Salary Prediction")
plt.xlabel("Employee Level")
plt.ylabel("Salary")
plt.legend()
plt.show()
```

## Output:
<img width="582" height="708" alt="image" src="https://github.com/user-attachments/assets/afd0fa53-0863-46d6-ae56-3086f5ef3eff" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
