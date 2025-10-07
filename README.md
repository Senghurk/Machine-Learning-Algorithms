# Machine Learning Projects Portfolio

A collection of machine learning algorithms implemented from scratch in Python, demonstrating fundamental ML concepts without relying on high-level libraries.

## Projects Overview

### 1. Genetic Algorithm Solver
**File:** `Genetic-Algorithm/GA.py`
- Solves linear equation: a + 2b + 3c + 4d = 30
- Implements roulette wheel selection for parent selection
- Features crossover and mutation operators
- Interactive console for experimenting with genetic operations
- Tracks fitness scores and convergence across generations

### 2. Multiple Linear Regression
**File:** `Multiple-Regression/Multiple-Regression.py`
- Built using NumPy for matrix operations
- Predicts BMI categories from height and weight
- Manually computes regression coefficients (b₀, b₁, b₂)
- No sklearn or other ML libraries used
- Includes BMI dataset for training

### 3. Multi-Layer Perceptron (MLP)
**File:** `MLP/Multi-Layer-Perceptron.py`
- 3-5-1 architecture neural network
- Sigmoid activation function
- Manual backpropagation implementation
- Gradient calculation and weight updates from scratch
- Trains on BMI prediction task

### 4. Logic Gate Classification
**Files:**
- `Logic-Gate-Classification/2input-AND-Function.py`
- `Logic-Gate-Classification/2input-OR-Function.py`
- `Logic-Gate-Classification/Sigmoid-Function.py`

- Simple perceptron implementation
- Delta learning rule for weight updates
- Step and sigmoid activation functions
- Models AND/OR gates behavior

## Key Features
- **From Scratch Implementation**: All algorithms coded without ML libraries
- **Educational Focus**: Clear code structure for learning purposes
- **Real-World Applications**: BMI prediction and classification tasks
- **Interactive Elements**: User input and experimentation options

## Requirements
```bash
numpy
pandas
```

## Usage
Each project can be run independently:

```bash
# Genetic Algorithm
python Genetic-Algorithm/GA.py

# Multiple Regression
python Multiple-Regression/Multiple-Regression.py

# MLP
python MLP/Multi-Layer-Perceptron.py

# Logic Gates
python Logic-Gate-Classification/2input-AND-Function.py
python Logic-Gate-Classification/2input-OR-Function.py
```

## Learning Outcomes
- Understanding of evolutionary algorithms
- Linear regression mathematics
- Neural network fundamentals
- Gradient descent and backpropagation
- Activation functions and their applications