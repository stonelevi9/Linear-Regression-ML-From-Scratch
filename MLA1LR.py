import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# This is our batch gradient descent method. It is responsible for getting the scalars of the linear regression (or
# the weights) closer to whatever the perfect scalars would be, to improve our linear regression's predictions. It
# works by first taking the shape of our training set and storing in variables m and n. Next we run a for loop that
# runs for the inputted number of epochs. We then predict a house value for each of our rows by taking the dot product
# of our training set and our weights as well as adding our b. This value is saved as predicted. Next we determine our
# error by subtracting our predicted house value from our actual house values and store it as error. Next we calculate
# the gradient in respect to our weights vector and store that as grad_w as well as the gradient in respect to our
# y-intercept and store that as grad_b. Last step is to update our weights and our y-intercept(b) using the gradients
# we calculated previously and our learning rate (lr). Then lastly, once the for loop is completed, we return our
# updated weights and y-intercept
def gradient_descent(train_x, train_y, weight, b, lr, epochs):
    m, n = train_x.shape
    for i in range(epochs):
        predicted = np.dot(train_x, weight) + b
        error = predicted - train_y
        grad_w = np.dot(train_x.T, error) / m
        grad_b = np.mean(error)
        weight = weight - lr * grad_w
        b = b - lr * grad_b
    return weight, b

# This is our predict method ,and it is responsible for actually predicting house values from our testing set after we
# have our (somewhat) optimal weights through our gradient descent. It works by first taking the shape of our testing
# set and saving its dimensions as q and r. We also create an array to store our predictions in as we generate them
# called predictions. Next we enter a for loop that runs for each row in our testing set and initialize our pred
# variable as 0(we will use this later). We then create a nested for loop that runs for each column of our current
# iteration and simply multiplies our predicted scalar by that columns value and sums all those columns*scalars up.
# We also add our y-intercept after adding all the other values and append the predicted value into our predictions
# array. Lastly once both for loops finish, we return our predictions array.
def predict(x_set, w, intercept):
    q, r = x_set.shape
    predictions = []
    for h in range(q):
        pred = 0
        for j in range(r):
            pred += x_set[h][j] * w[j]
        pred += intercept
        predictions.append(pred)
    return predictions

# This is our main method ,and firstly it just reads in the dataset as instructed. Then we split the data set into 4
# different sets using train_test_split. These four data sets are x_train (our training set inputs), x_test (our
# testing set inputs), y_train (our training set actual outputs), and y_test (our testing set actual outputs). We
# then normalize our x_train and x_test sets. We then initialize our weights array with some random integers as well
# as our y_int with a random integer. Next we call gradient descent and store it's output as weights and y_int. We
# then call our prediction method and store its output as prediction. Lastly we calculate our mean_squared_error and
# print this value.
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=.7)
x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)
weights = np.array([6, -4, 1, 6, 1, -5, 3, -1, 2, 1, -5, 1, -3])
y_int = 1
weights, y_int = gradient_descent(x_train, y_train, weights, y_int, 0.05, 10000)
prediction = predict(x_test, weights, y_int)
errors = mean_squared_error(y_test, prediction)
print("Mean Square Error: ", errors)
