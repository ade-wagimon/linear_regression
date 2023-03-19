# linear_regression

    # Import necessary libraries
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Create sample data
    x = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
    y = np.array([2, 4, 5, 4, 5])

    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the data
    model.fit(x, y)

    # Predict the output for new data points
    x_new = np.array([6, 7]).reshape((-1, 1))
    y_new = model.predict(x_new)

    # Print the predicted output
    print(y_new)


In this example, we first create some sample data consisting of input variable x and output variable y. We then create an instance of the LinearRegression class from scikit-learn and fit the model on the data using the fit() method.

Next, we predict the output for new data points using the predict() method and print the predicted output.

This is just a simple example to get you started with linear regression in Python. For more complex datasets, you may need to preprocess the data, perform feature engineering, and tune the hyperparameters of the model to achieve better performance.
