import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Define a function that can Train on Multiple variate Model for linear regression
def plot_learning_curves(model, X, y):
    # Start splitting the dataset into Train / Test Portion
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # Initialize the list for training errors and MSE
    train_errors, val_errors = [], []
    # Start the training for the whole length of the training portion
    for m in range(1, len(X_train)):
        # fit the data into the model (Training)
        model.fit(X_train[:m], y_train[:m])
        # Start predicting on the model using the testing portion
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        # Calculate the error of the model / cost / MSE
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    # Plot the model error
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

# Generate random feature of 100 samples
X = 6 * np.random.rand(100, 1) - 3
# Generate the Y target from the feature as a polynomial feature of degree 2
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
# Initialize the model to be passed into the function
lin_reg = LinearRegression()
# Start Training and Plot
plot_learning_curves(lin_reg, X, y)

plt.show()

