# coding:utf-8
# Author: Lg
# Date: 19/11/29

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FitnessFunc import FitnessFunction, FitnessFunction2, FeatureSelection
from DiffEvolution import DE
from fusionWork import FeatureFusion
from alpha_index import AlphaIndex


class Main(object):
    def __init__(self):
        data_path = 'D:/BraTS2019_Experiments/DataSets/seg_model/features/xgb_excel_normed_10.xlsx'
        # data_path = 'D:/BraTS2019_Experiments/DataSets/seg_model/features/test.xlsx'
        data = pd.read_excel(data_path)
        target = 'zLabel'
        predictors = [x for x in data.columns if x not in [target]]
        self.x_data = data[predictors]
        self.y_data = data[target]

        self.featureSelection = FeatureSelection()
        #self.fitnessFunc = FitnessFunction()
        self.fitnessFunc2 = FitnessFunction2()

        self.dim = len(predictors) + 1
        self.size = 10
        self.iter_num = 30
        self.x_max = 1.0
        self.x_min = 0.0
        self.max_vel = 0.05
        
        self.alpha = 0.0

    def saveDEResult(self):
        data = self.featureSelection.data
        data = data.drop(['label'], axis=1)
        print(data)
        table = pd.concat(
            [data, pd.Series(self.y_data, index=data.index)], axis=1)
        table.columns = list(data.columns) + ['zLabel']
        print(table.columns)
        print('Selected features number is: {}'.format(len(table.columns)))
        table.to_excel('../DataSets/DE_result/DE_features.xlsx', index=False)
        return len(table.columns)

    def fitness_func(self, weights):
        weights = self.featureSelection.selectByThreshold(weights)
        data = self.featureSelection.getSelectedFeatures(self.x_data, weights)
        # fitness = self.fitnessFunc.fitness(data, self.y_data)
        # fitness += (len(self.featureSelection.data.columns) / (self.dim-1)) * 4
        # print('The fitness value: {}\n'.format(fitness))
        # return fitness 
        # fitness = sum([x*10 for x in weights])
        # return fitness
        self.data = data
        
        fitness = self.fitnessFunc2.getFitness(data, self.y_data, self.alpha)
        return fitness
        

    def doMain(self):
        de = DE(self.fitness_func,
                self.dim,
                self.size,
                self.iter_num,
                self.x_min,
                self.x_max)

        fit_var_list2, best_pos2 = de.doUpdate()
        # print('Best solution sets: {}'.format(best_pos2))
        print('Fitness result under best solution sets: {}'.format(
            fit_var_list2[-1]))
        plt.plot(np.linspace(0, self.iter_num, self.iter_num),
                 fit_var_list2, c="G", alpha=0.5, label="DE")
        plt.xlabel('iteration')
        plt.ylabel('fitness')
        plt.legend()
        plt.grid()
        plt.savefig('../DataSets/DE_result/DE_process.png')
        plt.show()

        features = self.saveDEResult()
        fitness = fit_var_list2[-1]
        alpha_index = AlphaIndex()
        error_rate = alpha_index.getErrorRate(self.data, self.y_data)
        alpha_index.toTxtFile([self.alpha, error_rate, features])


if __name__ == '__main__':
    main = Main()
    main.doMain()



    # featureFusion = FeatureFusion()
    # featureFusion.clusterMeasure()

