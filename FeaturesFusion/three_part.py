# _*_ encoding:utf-8 _*_
# Date: 2020/2/4
# Author: Lg
# calcuating the three parts of fitness: Error rate, class dinstinct, and feature dimension.

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


class DataSets(object):
    def __init__(self):
        self.raw_file_path = ''
        self.normed_file_path = '/Users/manfestain/Downloads/FeaureFusion/xgb_excel_normed_8.xlsx'
        self.fusion_file_path = './DEselect_result/101_2.8DE_select.xlsx'
        # self.fusion_file_path = './DEselect_result/DE_features_bk.xlsx'
        self.readFile()

    def readFile(self):
        data = pd.read_excel(self.normed_file_path)
        columns = data.columns
        print(len(columns))
        mixed_columns = [name for name in columns if 'zLabel' not in name]
        print(len(mixed_columns))
        cnn_columns = [name for name in mixed_columns if 'cnnf_' in name]
        print(len(cnn_columns))
        rad_columns = [name for name in mixed_columns if 'cnnf_' not in name]
        print(len(rad_columns))
        
        self.label = data['zLabel']
        self.cnn_features = data[cnn_columns]
        self.radiomics_features = data[rad_columns]
        self.selected_features = pd.read_excel(self.fusion_file_path)
        print(len(self.selected_features.columns))
        self.mixed_features = data[mixed_columns]

def getEuclidean(vector1, vector2):
    temp = np.array(vector1) - np.array(vector2)
    temp = temp * temp
    return np.sqrt(np.sum(temp))

class ThreePart(object):
    def __init__(self):
        pass

    def getErrotRate(self, data, labels):
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.5)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        # matrix = confusion_matrix(y_test, y_pred)
        # print(matrix)
        # # print(matrix.sum())
        # temp = matrix * np.eye(matrix.shape[0])
        # temp = temp.sum()
        # return float((matrix.sum() - temp) / matrix.sum())
        acc = accuracy_score(y_pred, y_test)
        # print(1 - acc)
        return 1 - acc

    def getClassDistinct(self, data, labels):
        average = [[], [], [], []]
        samples = [[], [], [], []]
        for l in range(len(labels)):
            index = int(labels[l])
            samples[index].append(data.loc[l])
        # for _, row in labels.iterrows():
        #     indexs = int(row['label'])
        #     samples[indexs].append(row)
        
        # for s in samples:
        #     print(len(s))
        print(data.shape)

        sum1 = 0.0
        for index, sample in enumerate(samples):
            average[index] = np.mean(sample, axis=0)
            # print(np.mean(sample, axis=0))
            temp = 0.0
            for s in sample:
                temp += np.linalg.norm(average[index] - s)
            temp /= len(sample)
            sum1 += temp

        # print(len(average))
        # print(len(average[0]))
        # print(len(average[2]))

        sum2 = 0.0
        for i in range(len(average)):
            for j in range(len(average)):
                sum2 += np.linalg.norm(average[i] - average[j])
        sum2 /= 2
        print('\n')
        return [sum1, sum2]

    def getFeatureDimension(self, data):
        return data.shape[1]

    def getDataFrame(self, data, labels):
        sums = self.getClassDistinct(data, labels)
        return [sums[0],
                sums[1],
                self.getErrotRate(data, labels)]

def main():
    datasets = DataSets()
    threeParts = ThreePart()
    columns = ['intra-class similarity', 'inter-class difference', 'Error rate']
    dataFrame = pd.DataFrame(columns=columns)
    print(datasets.cnn_features.shape)
    print(datasets.radiomics_features.shape)
    print(datasets.mixed_features.shape)
    print(datasets.selected_features.shape)
    dataFrame.loc[0] = threeParts.getDataFrame(datasets.cnn_features, datasets.label)
    dataFrame.loc[1] = threeParts.getDataFrame(datasets.radiomics_features, datasets.label)
    dataFrame.loc[2] = threeParts.getDataFrame(datasets.mixed_features, datasets.label)
    dataFrame.loc[3] = threeParts.getDataFrame(datasets.selected_features, datasets.label)
    dataFrame.insert(0, 'func', ['cnn', 'radiomics', 'mixed', 'selected'])
    dataFrame.to_csv('./temp/three_parts_result.csv', index=False)


if __name__ == '__main__':
    main()
    # a = [1, 2, 3]
    # b = [4, 5, 6]
    # print(getEuclidean(a, b))
