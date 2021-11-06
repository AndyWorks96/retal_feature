# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/6/25
'''
    分割结果的评价指标：
        1. Dice coefficent
        2. XXX
        3. XXX
'''
import os
import cv2
import pandas as pd
import numpy as np 
import scipy.misc

class EvalutionIndex(object):
    def __init__(self):
        pass

    # DC指标
    def getDiceCoefficent(self, ground_truth, segmentation, label_idx=0):
        # [gh, gw] = ground_truth.reshape((240, 240))
        # [sh, sw] = segmentation.reshape((240, 240))
        ground_truth = (ground_truth==label_idx)
        segmentation = (segmentation==label_idx)
        # print(ground_truth)
        # print('\n')
        # print(segmentation)
        gh, sh, gw, sw = 1, 1, 1, 1
        if (gh==sh and gw==sw):
            prod = np.multiply(segmentation, ground_truth)
            s0 = prod.sum()
            s1 = segmentation.sum()
            s2 = ground_truth.sum()
            dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
            return dice
        return 0


    # Sensitivity指标
    def getSensitivity(self, ground_truth, segmentation, label_idx=0):
        # [gh, gw] = ground_truth.reshape(240, 240).shape()
        # [sh, sw] = segmentation.reshape(240, 240).shape()
        ground_truth = (ground_truth==label_idx)
        segmentation = (segmentation==label_idx)
        # print(ground_truth)
        # print('\n')
        # print(segmentation)
        gh, sh, gw, sw = 1, 1, 1, 1
        if (gh==sh and gw==sw):
            prod = np.multiply(segmentation, ground_truth)
            s0 = prod.sum()
            s1 = ground_truth.sum()
            dice = (s0 + 1e-10)/(s1 + 1e-10)
            return dice
        return 0


    # Specificity指标
    def getSpecificity(self, ground_truth, segmentation, label_idx=0):
        # [gh, gw] = ground_truth.reshape(240, 240).shape()
        # [sh, sw] = segmentation.reshape(240, 240).shape()
        ground_truth = (ground_truth==label_idx)
        segmentation = (segmentation==label_idx)
        gh, sh, gw, sw = 1, 1, 1, 1
        if (gh==sh and gw==sw):
            s0 = np.sum(np.multiply(segmentation==0, ground_truth==0))
            s1 = np.sum(ground_truth==0)
            dice = (s0 + 1e-10)/(s1 + 1e-10)
            return dice
        return 0


class SaveEvaltuion(object):
    def __init__(self):
        columns = ['name', 'dice_1', 'dice_2', 'dice_4', 'sen_1', 'sen_2', 'sen_4', 'spe_1', 'spe_2', 'spe_4']
        dataFrame = pd.DataFrame(columns=columns)
        self.idx = 0
        self.coumns = columns
        self.dataFrame = dataFrame

    # roi = [roi_1, roi_2, roi_4], roi_x = [dice, sensitivity, specifity]
    def addSeries(self, name, roi):
        # print(roi)
        data = [name]
        for r in roi:
            data.extend(r)
        self.dataFrame.loc[self.idx] = data
        self.idx += 1
                
    def save(self, save_path):
        dataFrame = self.dataFrame
        dataFrame.to_csv(save_path, index=False)
        print('Save to >{}< finished!'.format(save_path))


if __name__ == '__main__':
    # ei = EvalutionIndex()
    # g = np.array([[0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0],
    #               [0, 1, 1, 1, 1],
    #               [0, 0, 2, 2, 0],
    #               [0, 0, 2, 1, 0]])
    # s = np.array([[0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0],
    #               [0, 1, 1, 1, 1],
    #               [0, 2, 2, 2, 0],
    #               [0, 2, 2, 2, 0]])
    # dice = ei.getDiceCoefficent(g, s, 2)
    # print(dice)
    # sen = ei.getSensitivity(g, s, 0)
    # print(sen)
    # spe = ei.getSpecificity(g, s, 1)
    # print(spe)
    name1 = 'name1'
    roi1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    name2 = 'name2'
    roi2 = [[4, 5, 6], [4, 5, 6], [4, 5, 6]]
    
    se = SaveEvaltuion()
    se.addSeries(name1, roi1)
    se.addSeries(name2, roi2)
    se.save('./temp/dasdfasdf.csv')
    # dice = [1, 2, 3]
    # sen = [4, 5, 6]
    # spe = [7, 8, 9]
    # sen.extend(spe)
    # dice.extend(sen)
    # print(dice)

