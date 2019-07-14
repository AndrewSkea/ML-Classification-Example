from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from tools.funcs import print_cm


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataset = pd.read_csv('./data/heart.csv', sep=",")
targets = dataset['target']
data = dataset.drop('target', axis=1)
dataset.head()

print("\nThe number of rows per target category:\n{}".format(dataset.groupby('target').count()['sex']))
print("\nThis is the mean values for each column in each category:\n")
print(dataset.groupby('target').min())

# SCATTER PLOT MATRIX
pd.plotting.scatter_matrix(dataset[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']], figsize=(12, 10), diagonal='kde')
plt.show()

# HEATMAP
corr = dataset.loc[:, :].corr()
plt.imshow(corr)
plt.show()
kf = KFold(5, shuffle=True)

def get_kmeans_avg(classifier):
    means = []
    for train, test in kf.split(dataset):
        classifier.fit(data.loc[train], targets.loc[train])
        prediction = classifier.predict(data.loc[test])
        cur_mean = np.mean(prediction == targets.loc[test])
        means.append(cur_mean)
    return np.mean(means)


max_mean = (0, 0)
for n in range(1, 25):
    classifier = KNeighborsClassifier(n_neighbors=n)
    mean = get_kmeans_avg(classifier)
    max_mean = (n, mean) if mean > max_mean[1] else max_mean

print("\nNum neighbours: {}\t\tMax mean: {:.2f}%".format(max_mean[0], 100*max_mean[1]))
