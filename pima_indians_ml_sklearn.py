# Pima Indians dataset diabetes classification using euclidean distance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Read dataset
df = pd.read_csv("diabetes.csv")
train, test = train_test_split(df, test_size = 0.2)
lr = LogisticRegression(max_iter = 1000)
# separate data and outcomes
train_out = train['Outcome']
# Remove column from matrix
train_feat = train.drop(columns = ['Outcome'])#train.iloc[:,:8]
test_feat = test.iloc[:,:8]
test_out = test['Outcome']
lr.fit(train_feat, train_out)
print("Accuracy of model for training set ")
print(lr.score(train_feat, train_out))
print("Accuracy of model for test set ")
print(lr.score(test_feat, test_out))
