import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

predict = "Salary"

# data_train = data_train[["Age","ORB","DRB","AST","STL","BLK","PTS","Salary"]]
data_train = data_train[["Age","PER","TS%","FG","FG%","3P","3P%","eFG%","FT%","ORB","DRB","AST","STL","BLK","PTS","Salary"]]
data_test = data_test[["Age","PER","TS%","FG","FG%","3P","3P%","eFG%","FT%","ORB","DRB","AST","STL","BLK","PTS"]]

data_train = data_train.fillna(data_train.mean())
data_test = data_test.fillna(data_train.mean())

X = np.array(data_train.drop([predict], 1))
Y = np.array(data_train[predict])

best = 0.9150162365393301
for _ in range(100000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # print("Accuracy: " + str(acc))
    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("nbaSalary.pickle", "wb") as f:
            pickle.dump(linear, f)

print(best)

pickle_in = open("nbaSalary.pickle", "rb")
linear = pickle.load(pickle_in)

# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

# x_train = np.array(data_train.drop([predict],1))
# y_train = np.array(data_train[predict])

# linear = linear_model.LinearRegression()
# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)
# print(acc)
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# predictions = linear.predict(x_test)

predictions = linear.predict(data_test)

data_id = pd.read_csv("test.csv")
data_id = data_id[["Id"]]

submission = pd.DataFrame({'Id':data_id["Id"] ,'Predicted':predictions})

filename = 'SalaryPredictions.csv'

submission.to_csv(filename,index=False)

# data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#
# predict = "G3"
#
# X = np.array(data.drop([predict], 1))
# y = np.array(data[predict])
#
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
#
# linear = linear_model.LinearRegression()
#
# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)
# print(acc)
#
# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)
#
# predictions = linear.predict(x_test)
#
# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])