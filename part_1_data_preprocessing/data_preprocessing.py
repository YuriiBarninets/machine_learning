# Import data set from Data.csv file
import pandas
dataset = pandas.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Missing data :
# Calculate mean for Age and Salary columns
from sklearn.preprocessing import Imputer
imputer = Imputer("NaN", "mean", 0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Categorical data :
# Use LabelEncoder to transform Country and Purchased values
# in range between 0 and n_classes-1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:, 0] = labelEncoder.fit_transform(X[:, 0])
Y = labelEncoder.fit_transform(Y)

# Categorical data :
# Use One-hot(aka One-of-k) data Encoder 
# to encode Country integer features in one-of-K scheme
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)