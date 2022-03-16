import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

draft = pd.read_csv('./Speed-Agility-Draft.csv')

def euclideanDist(x, sample):
    return np.sqrt(np.power(x-sample, 2).sum(axis=1)).sort_values(ascending=True)

def manhattanDist(x, sample):
    return np.absolute(x-sample).sum(axis=1).sort_values(ascending=True)

def knn(x, y, sample, k, distance):
    pred = []
    for i in sample.index:
        if (distance == 'Euclidean'):
            dist = euclideanDist(x, sample.iloc[i])
        elif (distance == 'Manhattan'):
            dist = manhattanDist(x, sample.iloc[i])
        else:
            return 0
        classification = y.iloc[dist[0:3].index].value_counts().reset_index().iloc[0,0]
        pred.append(classification)
    return pred

x = draft[['Speed', 'Agility']]
y = draft[['Draft']]

sample = pd.DataFrame([[6.75, 3], [5.34, 6], [4.67, 8.4], [7, 7], [7.8, 5.4]], columns = ['Speed', 'Agility'])

k = 3
sample['3_Euclidean'] = knn(x, y, sample, k, 'Euclidean')
sample['3_Manhattan'] = knn(x, y, sample, k, 'Manhattan')

k = 5
sample['5_Euclidean'] = knn(x, y, sample, k, 'Euclidean')
sample['5_Manhattan'] = knn(x, y, sample, k, 'Manhattan')

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1,1,1)
ax = sns.scatterplot(x = 'Speed', y = 'Agility',hue = 'Draft', s=50, data = draft)
ax = sns.scatterplot(x = 'Speed', y = 'Agility', hue = '3_Euclidean',marker = "X", s=200, data = sample)
plt.show()
