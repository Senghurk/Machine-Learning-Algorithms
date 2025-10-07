"""
Multiple Linear Regression Implementation from Scratch
Predicts BMI categories from height and weight using manually computed coefficients
"""

import pandas as pd
import numpy as np

def calculate_sums(X, y):
    n = len(X)
    X1, X2 = X[:, 0], X[:, 1]
    
    sum_X1 = np.sum(X1)
    sum_X2 = np.sum(X2)
    sum_y = np.sum(y)
    sum_X1_squared = np.sum(X1**2)
    sum_X2_squared = np.sum(X2**2)
    sum_X1y = np.sum(X1 * y)
    sum_X2y = np.sum(X2 * y)
    sum_X1X2 = np.sum(X1 * X2)
    
    return sum_X1, sum_X2, sum_y, sum_X1_squared, sum_X2_squared, sum_X1y, sum_X2y, sum_X1X2, n

def calculate_regression_coefficients(X, y):
    sum_X1, sum_X2, sum_y, sum_X1_squared, sum_X2_squared, sum_X1y, sum_X2y, sum_X1X2, n = calculate_sums(X, y)
    
    reg_sum_X1 = sum_X1_squared - (sum_X1**2 / n)
    reg_sum_X2 = sum_X2_squared - (sum_X2**2 / n)
    reg_sum_y = np.sum(y**2) - (sum_y**2 / n)
    reg_sum_X1y = sum_X1y - (sum_X1 * sum_y / n)
    reg_sum_X2y = sum_X2y - (sum_X2 * sum_y / n)
    reg_sum_X1X2 = sum_X1X2 - (sum_X1 * sum_X2 / n)
    
    denominator = reg_sum_X1 * reg_sum_X2 - reg_sum_X1X2**2
    
    
    b1 = (reg_sum_X2 * reg_sum_X1y - reg_sum_X1X2 * reg_sum_X2y) / denominator
    
    b2 = (reg_sum_X1 * reg_sum_X2y - reg_sum_X1X2 * reg_sum_X1y) / denominator
    
    b0 = (sum_y / n) - b1 * (sum_X1 / n) - b2 * (sum_X2 / n)


    return b0, b1, b2

def predict_bmi_category(x1, x2, b0, b1, b2):
    y_hat = b0 + b1 * x1 + b2 * x2
    return round(y_hat)  

data = pd.read_csv('bmi.csv', header=None)
X = data[[1, 2]].values
y = data[3].values

b0, b1, b2 = calculate_regression_coefficients(X, y)

print(f"Estimated Linear Regression Equation:")
print(f"BMI Category = {b0:.3f} + {b1:.3f} * Height + {b2:.3f} * Weight")

while True:
    try:
        height = float(input("Enter height in cm: "))
        weight = float(input("Enter weight in kg: "))
        break
    except ValueError:
        print("Please enter valid numbers for height and weight.")

predicted_category = predict_bmi_category(height, weight, b0, b1, b2)

bmi_categories = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

print(f"\nPredicted BMI category for [height = {height}cm] and [weight = {weight}kg]:\n")
print(f"{predicted_category} - {bmi_categories.get(predicted_category, 'Unknown Category')}")