import numpy as np

x1 = 0.3
x2 = 0.4
y = 0.6
alpha = 0.1

w1 = 0.5
w2 = 0.5
b = 0.5

for iteration in range(1, 11):
    print(f"Iteration {iteration}:")
    
    # Calculate z
    z = w1 * x1 + w2 * x2 + b
    
    # Sigmoid
    y_prime = 1 / (1 + np.exp(-z))
    
    # Error
    e = y - y_prime
    
    # Error gradient (δ)
    delta = y_prime * (1 - y_prime) * e
    
    # Calculate gradients
    grad_w1 = alpha * x1 * delta
    grad_w2 = alpha * x2 * delta
    grad_b = alpha * delta  
        
    # Update weights and bias
    w1 = w1 + grad_w1
    w2 = w2 + grad_w2
    b = b + grad_b
    
    print(f"Output: {y_prime:.6f}, Error: {e:.6f}")
    print(f"w1: {w1:.6f}, w2: {w2:.6f}, b: {b:.6f}")
    print()
