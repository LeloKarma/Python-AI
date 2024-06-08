import numpy as np
import tkinter as tk

# OR Gate inputs and outputs
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

outputs = np.array([0, 1, 1, 1])

# Initial weights
w1 = 0.3
w2 = -0.3

# Threshold (activation)
theta = 0.2

# Learning rate
alpha = 0.1

# Function to calculate the actual output
def calculate_output(x1, x2, w1, w2, theta):
    z = x1 * w1 + x2 * w2
    return 1 if z >= theta else 0

# Perceptron learning algorithm
converged = False
epoch = 0

while not converged:
    converged = True
    for i in range(len(inputs)):
        x1, x2 = inputs[i]
        Y_d = outputs[i]
        Y = calculate_output(x1, x2, w1, w2, theta)
        error = Y_d - Y

        if error != 0:
            converged = False
            w1 = w1 + alpha * error * x1
            w2 = w2 + alpha * error * x2

        print(f"Epoch: {epoch}, Input: ({x1}, {x2}), Desired Output: {Y_d}, Actual Output: {Y}, Error: {error}, Weights: ({w1}, {w2})")

    epoch += 1

print(f"Final Weights: w1 = {w1}, w2 = {w2} after {epoch} epochs")

