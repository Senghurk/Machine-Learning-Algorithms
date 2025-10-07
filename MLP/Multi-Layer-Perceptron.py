"""
Multi-Layer Perceptron (MLP) Implementation from Scratch
1-layer neural network with sigmoid activation and manual backpropagation
Includes weight updates and gradient calculation for BMI prediction
"""

import numpy as np
import pandas as pd

class MLP:
    def __init__(self, learning_rate=0.01):
        np.random.seed(42)
        self.learning_rate = learning_rate
        
        # 3-5-1 
        self.input_to_hidden_weights = np.random.randn(3, 5) * 0.5 # 15
        self.hidden_bias = np.zeros(5) # 5
        self.hidden_to_output_weights = np.random.randn(5, 1) * 0.5 # 5
        self.output_bias = 0.0 # 1
        
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(self, input_data):
        # Input to Hidden layer 
        hidden_layer_input = np.zeros((input_data.shape[0], 5))
        for i in range(input_data.shape[0]):  
            for j in range(5):  
                hidden_layer_input[i, j] = 0
                for k in range(3):  
                    hidden_layer_input[i, j] += input_data[i, k] * self.input_to_hidden_weights[k, j]
                hidden_layer_input[i, j] += self.hidden_bias[j]
        
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        
        # Hidden to Output layer 
        output_layer_input = np.zeros(input_data.shape[0])
        for i in range(input_data.shape[0]):  
            output_layer_input[i] = 0
            for j in range(5):  
                output_layer_input[i] += hidden_layer_output[i, j] * self.hidden_to_output_weights[j, 0]
            output_layer_input[i] += self.output_bias
        
        output_layer_output = self.sigmoid(output_layer_input)
        return hidden_layer_output, output_layer_output
    
    def train(self, input_features, target_bmi, epochs=1000):
        target_bmi = target_bmi.flatten()
        
        for epoch in range(epochs):
            hidden_layer_output, predicted_bmi = self.forward(input_features)
            
            # Calculate errors
            prediction_error = predicted_bmi - target_bmi
            sum_squared_error = np.sum(prediction_error ** 2)
            mean_loss = sum_squared_error / len(input_features)
            accuracy = 100 - (np.mean(np.abs(prediction_error)) * 100)
            
            # Backpropagation
            # Output layer gradients
            output_delta = prediction_error * predicted_bmi * (1 - predicted_bmi)
            
            # Hidden layer gradients
            hidden_error = output_delta.reshape(-1, 1) * self.hidden_to_output_weights.T
            hidden_delta = hidden_error * hidden_layer_output * (1 - hidden_layer_output)
            
            # Update weights 
            # Update hidden_to_output_weights
            for i in range(5):  
                weight_gradient = 0
                for j in range(len(input_features)):  
                    weight_gradient += hidden_layer_output[j, i] * output_delta[j]
                self.hidden_to_output_weights[i, 0] -= self.learning_rate * weight_gradient / len(input_features)
            
            # Update output_bias
            bias_gradient = 0
            for j in range(len(input_features)):
                bias_gradient += output_delta[j]
            self.output_bias -= self.learning_rate * bias_gradient / len(input_features)
            
            # Update input_to_hidden_weights
            for i in range(3):  
                for j in range(5):  
                    weight_gradient = 0
                    for k in range(len(input_features)):  
                        weight_gradient += input_features[k, i] * hidden_delta[k, j]
                    self.input_to_hidden_weights[i, j] -= self.learning_rate * weight_gradient / len(input_features)
            
            # Update hidden_bias
            for j in range(5):  
                bias_gradient = 0
                for k in range(len(input_features)):  
                    bias_gradient += hidden_delta[k, j]
                self.hidden_bias[j] -= self.learning_rate * bias_gradient / len(input_features)
            
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch: {epoch}")
                print(f"Target BMI: {target_bmi[:5]}")
                print(f"Predicted BMI: {predicted_bmi[:5]}")
                print(f"Prediction Error: {prediction_error[:5]}")
                print(f"Sum of Squared Error: {sum_squared_error:.6f}")
                print(f"Mean Loss: {mean_loss:.6f}")
                print(f"Accuracy: {accuracy:.2f}%")
                print("-" * 70)
    
    def predict(self, input_features):
        hidden_layer_output, predicted_bmi = self.forward(input_features)
        return predicted_bmi

# BMI dataset
bmi_dataset = pd.read_csv('bmi.csv', header=None)
input_features = bmi_dataset.iloc[:, :3].values  
target_bmi_values = bmi_dataset.iloc[:, 3].values  

# Train MLP model
bmi_predictor = MLP(learning_rate=0.1)
bmi_predictor.train(input_features, target_bmi_values, epochs=1000)

final_predictions = bmi_predictor.predict(input_features)
final_mean_squared_error = np.mean((target_bmi_values - final_predictions) ** 2)
print(f"Final Mean Squared Error: {final_mean_squared_error:.4f}")