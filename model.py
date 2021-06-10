#import libraries
import numpy as np
import pandas as pd
import pickle

#import dataset
data = pd.read_csv('E:\WebApp\IrisFlower\iris.csv')

data = data.drop(['Id'],axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

x = data.drop('Species',axis=1)
y = data['Species']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(f'Rows in train set: {len(x_train)}\nRows in test set: {len(x_test)}')

from sklearn.svm import SVC
svm_model = SVC()
classifier = svm_model.fit(x_train,y_train)

pickle.dump(classifier,open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
