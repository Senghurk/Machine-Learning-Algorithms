"""
Logic Gate Classification using Delta Learning Rule
Trains a simple perceptron to model OR gate using step function
"""

import numpy as np

# Data for OR function
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 1])

w1 = 0.3
w2 = -0.1
b = 0.2  
alpha = 0.1  #learning rate

def step_function(x):
    return 1 if x >= 0 else 0
print("\n[ 2-input OR functions ]\n")
print(f"{'Epoch':<6}{'x1':<4}{'x2':<4}{'Yd':<4}{'w1':>6}{'w2':>6}{'Y':>6}{'e':>4}{'w1':>6}{'w2':>6}")
print("-" * 56)

epochs = 5

for epoch in range(1, epochs + 1):
    for i in range(len(inputs)):
        x1, x2 = inputs[i]
        Yd = outputs[i]
        
        weighted_sum = w1 * x1 + w2 * x2 - b
        Y = step_function(weighted_sum)

        e = Yd - Y
        
        old_w1, old_w2 = w1, w2
        
        w1 = w1 + alpha * x1 * e
        w2 = w2 + alpha * x2 * e
        
        print(f"{epoch:<6}{x1:<4}{x2:<4}{Yd:<4}{old_w1:>6.1f}{old_w2:>6.1f}{Y:>6}{e:>4}{w1:>6.1f}{w2:>6.1f}")
    
    if epoch < epochs:
        print("-" * 56)

print(f"\nbias: b = {b}; learning rate: α = {alpha}")
print(f"Final weights: w1 = {w1:.1f}, w2 = {w2:.1f}")
