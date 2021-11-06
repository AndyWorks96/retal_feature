# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/11/29
# Evaluate the DE`s result:
#   chose feature firstly,
#   cluster the feature by k-means secondly,
#   construct the fitness function finally.

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class FeatureSelection(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def setThreshold(self, threshold):
        self.threshold = threshold

    def selectByThreshold(self, weights):
        self.setThreshold(weights[-1])
        threshold = self.threshold
        new_weights = []
        for i in range(len(weights) - 1):
            if weights[i] > threshold:
                new_weights.append(1)
            else:
                new_weights.append(0)
        return new_weights

    def getSelectedFeatures(self, data, weights):
        columns = data.columns
        new_columns = []
        for i in range(len(weights) - 1):
            if weights[i] == 0:
                new_columns.append(columns[i])
        new_data = data.drop(new_columns, axis=1)
        self.data = new_data
        print('The selected features numbers: {}'.format(len(new_data.columns)))
        return new_data


# 采用熵衡量K-means聚类结果的适应度函数
class FitnessFunction(object):
    def __init__(self):
        pass

    def doKMeans(self, data):
        y = KMeans(n_clusters=4).fit(data)
        # table = pd.concat([data, pd.Series(y.label_, index=data.index)], axis=1)
        # table.columns = list(data.columns) + ['Zlabel']
        return y.labels_

    def getEntropy(self, data):   # calcute the entropy in every label
        _, counts = np.unique(data, return_counts=True)
        lengths = len(data)
        frequency = [x/lengths for x in counts]
        ent = 0.0
        for i in range(len(frequency)):
            ent -= frequency[i] * math.log(frequency[i], 2)
        # print('Every Entropy: {}'.format(ent))
        return ent

    def getLabels(self, labels, GT, flag):
        new_labels = []
        for i in range(len(labels)):
            if labels[i] == flag:
                new_labels.append(GT[i])
        return self.getEntropy(new_labels)

    def fitness(self, data, GT):
        labels = list(self.doKMeans(data))
        result = 0.0
        for f in [0, 1, 2, 3]:
            # calcute the fintess function
            result += self.getLabels(labels, GT, f)
        return result


# 新的适应度函数：KNN分类结果 + 类内距离/类间距离 + 特征维度
class FitnessFunction2(object):
    def __init__(self):
        pass

    def errorRate(self, data, labels):
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        matrix = confusion_matrix(y_test, y_pred)
        # print(matrix
        # print(matrix.sum())
        temp = matrix * np.eye(matrix.shape[0])
        temp = temp.sum()
        # print(temp)

        return (matrix.sum() - temp) / matrix.sum()

    def classDistinct(self, data, labels):
        average = [0 for i in range(4)]
        samples = [[], [], [], []]
        data['label'] = labels
        for _, row in data.iterrows():
            indexs = int(row['label'])
            samples[indexs].append(row)

        sum1 = 0.0
        for index, sample in enumerate(samples):
            average[index] = np.mean(sample, axis=0)
            for s in sample:
                sum1 += np.linalg.norm(average[index] - s)

        sum2 = 0.0
        for i in range(len(average)):
            for j in range(len(average), i, -1):
                sum2 += np.linalg.norm(i - j)

        return sum1 / sum2

    def Dimension(self, data, alpha=0.0):
        return alpha * data.shape[1]

    def getFitness(self, data, GT, alpha):
        error_rate = self.errorRate(data, GT)
        distinct = self.classDistinct(data, GT)
        dimension = self.Dimension(data, alpha)
        # print('error rate: {}\nclass distinct: {}\nfeature dims: {}'.format(error_rate, distinct, dimension))
        return error_rate + distinct + dimension
