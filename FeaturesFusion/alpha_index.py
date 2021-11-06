# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 20/1/14
# 对参数alpha的影响进行分析

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

class AlphaIndex(object):
    def __init__(self):
        self.txt_file = '../DataSets/DE_result/alpha_records.txt'

    
    def toTxtFile(self, msg):
        '''
        msg = [xx, xx, xx]
        alpha_records.txt: alpha\tErrorRate\tFeature
        '''
        messages = ''
        for m in msg:
            messages += str(m) 
            messages += '\t'
        with open(self.txt_file, 'a') as f:
            f.write(messages)

    def getErrorRate(self, data, labels):
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        matrix = confusion_matrix(y_test, y_pred)
        temp = matrix * np.eye(matrix.shape[0])
        temp = temp.sum()

        return (matrix.sum() - temp) / matrix.sum()

    def plotAlpha(self):
        alphas = []
        error_rates = []
        features = []
        with open(self.txt_file) as f:
            for line in f:
                strs = line.split('\t')
                alphas.append(strs[0])
                errot_rates.append(strs[1])
                features.append(strs[2])
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, error_rates, 'ob')
        ax1.plot(x, error_rates, 'b', label='Error Rate')
        ax1.set_ylabel('Error Rate/%')
        ax1.legend(loc=2)
        plt.ylim([0.0, 1.0])
        ax1.set_xlabel("alpha")

        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(x, features, 'or')
        ax2.plot(x, features, 'r', label='Feature dimension')
        ax2.set_ylabel('Feature dimension')
        ax2.legend(loc=1)
        plt.ylim([0, 550])

        plt.savefig('./temp/alpha_index.png', dpi=500)
        # plt.show()

if __name__ == '__main__':
    alpha_index = AlphaIndex()
    msg = [0.3, 120, 0.75]
    alpha_index.toTxtFile(msg)
