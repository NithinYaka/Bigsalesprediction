import numpy as np

# Generate some sample data
# Replace this with your actual sales data
# For simplicity, I'm using random data here
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Define the learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Initialize the parameters
theta = np.random.randn(2, 1)

# Add a bias term (x0 = 1) to the input features
X_b = np.c_[np.ones((100, 1)), X]

# Perform gradient descent
for iteration in range(num_iterations):
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

# Print the learned parameters
print("Learned Parameters:")
print("Theta_0 (bias):", theta[0][0])
print("Theta_1 (slope):", theta[1][0])

# Now you can make predictions
# For example, predict the sales for a new input value of X
new_X = np.array([[2]])  # Replace with your desired input value
new_X_b = np.c_[np.ones((1, 1)), new_X]
predicted_sales = new_X_b.dot(theta)
print("Predicted Sales:", predicted_sales[0][0])
