import numpy as np
import pandas as pd

# Load and preprocess the data
data = pd.read_csv('DATA/mnist_train.csv') #Specify the path to the dataset here

# Convert data to numpy array and shuffle
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Split data into training and development sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]  # Labels for development set
X_dev = data_dev[1:n] / 255.0  # Normalize features for development set

data_train = data[1000:m].T
Y_train = data_train[0]  # Labels for training set
X_train = data_train[1:n] / 255.0  # Normalize features for training set

print(Y_train)  # Print training labels for verification

# Initialize parameters for a neural network with one hidden layer and Leaky ReLU activation
def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(2 / 784)  # He initialization
    b1 = np.random.randn(10, 1) * 0.01  # Small random values
    W2 = np.random.randn(10, 10) * np.sqrt(2 / 10)  # He initialization
    b2 = np.random.randn(10, 1) * 0.01  # Small random values
    return W1, b1, W2, b2

# Leaky ReLU activation function
def leaky_ReLU(Z, alpha=0.01):
    return np.maximum(alpha * Z, Z)

# Derivative of Leaky ReLU
def deriv_leaky_ReLU(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha)

# Softmax activation function with numerical stability
def softmax(Z):
    Z_exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return Z_exp / np.sum(Z_exp, axis=0, keepdims=True)

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1  # Linear transformation for hidden layer
    A1 = leaky_ReLU(Z1)  # Activation for hidden layer
    Z2 = W2.dot(A1) + b2  # Linear transformation for output layer
    A2 = softmax(Z2)  # Activation for output layer
    return Z1, A1, Z2, A2

# One-hot encoding for labels
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# Backward propagation
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size  # Number of examples
    one_hot_Y = one_hot(Y)  # Convert labels to one-hot format
    
    # Gradients for output layer
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    # Gradients for hidden layer
    dZ1 = W2.T.dot(dZ2) * deriv_leaky_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Update parameters using gradient descent
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

# Get predictions from the output of the network
def get_predictions(A2):
    return np.argmax(A2, axis=0)

# Calculate the accuracy of predictions
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent to train the network
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()  # Initialize parameters
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)  # Forward propagation
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)  # Backward propagation
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)  # Update parameters

        # Print progress every 10 iterations
        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}, Accuracy: {accuracy:.4f}")

    return W1, b1, W2, b2

# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=10000, alpha=0.1)