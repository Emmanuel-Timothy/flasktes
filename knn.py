import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
x = df.iloc[:, 1:5].values
y = df.iloc[:, 5].values

df["Species"] = df['Species'].astype('category')

df['Species'] = df['Species'].cat.codes
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.3, random_state=42, shuffle=False)
knn = KNeighborsClassifier(n_neighbors=25)
model = knn.fit(x_train,y_train)

pickle.dump(model, open('knn.pkl', 'wb'))