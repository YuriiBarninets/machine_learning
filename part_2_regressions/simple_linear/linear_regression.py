# Import data set
import pandas
import matplotlib.pyplot as plt

dataset = pandas.read_csv("salary_data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Split data on training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting results by using regressor trained on the Training set
Y_predicted_test = regressor.predict(X_test)

# Visualize results
plt.scatter(X_train, Y_train, color = 'red')
plt.scatter(X_test, Y_test, color = 'orange')
plt.plot(X_test, Y_predicted_test, color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()