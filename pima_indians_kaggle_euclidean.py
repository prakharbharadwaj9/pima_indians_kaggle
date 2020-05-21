import pandas as pd
import random
import numpy,scipy
from scipy.spatial import distance
# Read dataset

user_data = pd.read_csv("diabetes.csv")
#print(user_data.columns)
#print(user_data.iloc[0] + user_data.iloc[1])
# Randomly select 460 samples for training model

# Algo = ML - classification or non-ML - euclidean distance
# Euclidean distance - find centroid of  0 and 1 parts of dataset
i0 = 0
i1 = 0
d0 = 0
d1 = 0
for i in range(540):
    if user_data.iloc[i,8] == 0:
        d0 = d0 + user_data.iloc[i] #iloc stamds for integer location, allows
        #accessing a row directly
        i0 += 1
    else:
        d1 = d1 + user_data.iloc[i]
        i1 += 1

# calculate centroids of first 460 data points
cent0 = d0/i0
cent1 = d1/i1

#print(cent0)
#print(cent1)
#print(f"No of negatives {i0}")
#print(f"No of positives {i1}")
# select 1 sample and measure euclidean distance from both centroids classify
cat = 2
right = 0
wrong = 0
for j in range(540,651):
    temp = user_data.iloc[j]
    dist0 = distance.euclidean(cent0[0:8],temp[0:8])
    dist1 = distance.euclidean(cent1[0:8],temp[0:8])
    # whichever is closer
    if dist0<=dist1:
        cat = 0
    else:
        cat = 1
# compare with actual value - find correct percentage - PERFORMANCE MEASURE
    if cat == user_data.iloc[j,8]:
        right += 1
    else:
        wrong += 1

error = (wrong/(right + wrong))*100
accuracy = (right/(right + wrong))*100

print(f"accuracy of model on test set {accuracy}%")
print(f"error of model of test set {error}%")

# validation set
cat = 0
right = 0
wrong = 0
for j in range(651,768):
    temp = user_data.iloc[j]
    dist0 = distance.euclidean(cent0[0:8],temp[0:8])
    dist1 = distance.euclidean(cent1[0:8],temp[0:8])
    # whichever is closer
    if dist0<=dist1:
        cat = 0
    else:
        cat = 1
# compare with actual value - find correct percentage - PERFORMANCE MEASURE
    if cat == user_data.iloc[j,8]:
        right += 1
    else:
        wrong += 1

error = (wrong/(right + wrong))*100
accuracy = (right/(right + wrong))*100

print(f"accuracy of model on validation set {accuracy}%")
print(f"error of model of validation set {error}%")
