import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


with open('names') as f_names:
    names = f_names.read().rstrip('\n').split('\t')


cars_train = pd.read_csv('train/train.tsv', sep='\t', names=names)

cars_train = pd.get_dummies(cars_train, columns=['engineType'])


y_train = pd.DataFrame(cars_train['price'])
cars_train.drop('price', inplace=True, axis=1)
cars_train.drop('brand', inplace=True, axis=1)
x_train = pd.DataFrame(cars_train)

model = LinearRegression()
model.fit(x_train, y_train)

names.remove('price')

cars_dev = pd.read_csv('dev-0/in.tsv', sep='\t', names=names)
with open('dev-0/expected.tsv', 'r') as dev_exp_f:
    Y_dev = np.array([float(x.rstrip('\n')) for x in dev_exp_f.readlines()])

cars_dev = pd.get_dummies(cars_dev, columns=['engineType'])
cars_dev.drop('brand', inplace=True, axis=1)
X_dev = pd.DataFrame(cars_dev)

Y_dev_predicted = model.predict(X_dev)
print(Y_dev_predicted)
pd.DataFrame(Y_dev_predicted).to_csv('dev-0/out.tsv', sep='\t', index=False, header=False)


cars_test=pd.read_csv('test-A/in.tsv', sep='\t', names=names)
cars_test = pd.get_dummies(cars_test, columns=['engineType'])
cars_test.drop('brand', inplace=True, axis=1)
X_test = pd.DataFrame(cars_test)

Y_test_predicted = model.predict(X_test)
pd.DataFrame(Y_test_predicted).to_csv('test-A/out.tsv', sep='\t', index=False, header=False)

error = np.sqrt(mean_squared_error(Y_dev, Y_dev_predicted))
print(error)
