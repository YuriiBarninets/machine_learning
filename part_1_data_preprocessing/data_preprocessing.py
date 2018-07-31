# import data set from Data.csv file
import pandas
dataset = pandas.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# calculate mean for missing data
from sklearn.preprocessing import Imputer
imputer = Imputer("NaN", "mean", 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])