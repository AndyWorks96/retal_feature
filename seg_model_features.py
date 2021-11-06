# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/7/23

'''
    对通过CNN提取的特征和Radiomics提取的特征进行特征处理：
    1. 特征融合
    2. 特征提取
'''

import os
import cv2
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk 
import matplotlib.pyplot as plt 
from glob import glob
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from FeatureExtraction import FirstOrderStatistics, TexturalFeatures, WaveletFeatures, CNNFeatures
from seg_data_library import SegDataLibrary
from file_path import seg_features_, seg_data_library_

# model_save_dir = './Runed_Result/ModelWeights/Features_model/'

class SegModelFeatures(object):
    def __inti__(self):
        self.fos = FirstOrderStatistics.FirstOrderStatistics()
        self.glcm = TexturalFeatures.GLCM()
        self.glrlm = TexturalFeatures.GLRLM()
        self.hog = TexturalFeatures.HOG()
        self.lbp = TexturalFeatures.LBP()
        self.cnn = CNNFeatures.CNNFeatures()

        sf_ = seg_features_()
        self.l1 = joblib.load(sf_.l1_path)
        self.l1l2 = joblib.load(sf_.l1l2_path)
        self.pca = joblib.load(sf_.pca_path)


    # 影像组学特征提取
    def getRadiomicsFeatures(self, arrays, idxs=[0, 1, 2, 3]):
        all_features = {}
        shuffix = ['f', 't1', 't1c', 't2']
        for i in idxs:
            arr = arrays[i]
            fos_f = self.fos.getFirstOrderStatistics(arr)
            glcm_f = self.glcm.getGLCMFeatures(arr.astype(np.int64))
            glrlm_f = self.glrlm.getGLRLMFeatures(arr.astype(np.int64))
            hog_f = self.hog.getHOGFeatures(arr)
            lbp_f = self.hog.getHOGFeatures(self.lbp.getLBPFeatures(arr), name='lbp')
            features = {**fos_f, **glcm_f, **glrlm_f, **hog_f, **lbp_f}
            tmp_features = {}
            for key in features:
                new_key = key + '_' + shuffix[i]
                tmp_features[new_key] = features[key]
            all_features = {**all_features, ** tmp_features}
        return all_features


    # CNN特征提取
    def getCNNFeatures(self, arrays, idxs=[0, 1, 2, 3]):
        all_features = {}
        shuffix = ['f', 't1', 't1c', 't2']
        for i in idx:
            arr = arrays[i]
            cnn_f = self.cnn.getFeatures(arr)
            key = [str(s) + '_' + shuffix[i] for s in range(1, len(cnn_f))]
            feature = dict(zip(key, cnn_f))
            all_features = {**all_features, **features}
        return all_features


    # 特征融合
    def mergeFeatures(self, arrays, idx=[0, 1, 2, 3]):
        radiomics_f = self.getRadiomicsFeatures(arrays, idx)
        cnn_f = self.getCNNFeatures(arrays, idx)
        features = {**radiomics_f, **cnn_f}
        return features


    # 特征选择
    def featuresSelection(self, features, method='l1'):
        if method == 'l1':
            return self.l1.transform(features)
        elif method == 'l1l2':
            return self.l1l2.transform(features)
        else:
            return self.pca.tranform(features)



# 自定义L1&L2选择方法
class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, 
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter, 
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        for i in range(cntOfRow):   #权值系数矩阵的行数对应目标值的种类数目
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                if coef != 0:   #L1逻辑回归的权值系数不为0
                    idx = [j]
                    coef1 = self.l2.coef_[i][j]   #对应在L2逻辑回归中的权值系数
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:   #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                            idx.append(k)
                    mean = coef / len(idx)   #计算这一类特征的权值系数均值
                    self.coef_[i][idx] = mean
        return self

# L1&L2正则化选择
class L1L2FeatureSelection(object):
    def __init__(self, threshold=0.01, C=0.1, solver='livlinear', n_jobs=2, save_path='./temp/l1l2_model.h5'):
        self.save_path = save_path
        self.threshold = threshold
        self.select_model = SelectFromModel(LR(threshold=threshold, C=C))

    def fitModel(self, X, y):
        self.select_model.fit(X, y)
        joblib.dump(self.select_model, self.save_path)


# L1正则化选择
class L1FeaturesSelection(object):
    def __init__(self, threshold=0.01, C=0.1, solver='livlinear', n_jobs=2, save_path='./temp/l1_model.h5'):
        self.save_path = save_path
        self.base_est = LogisticRegression(penalty='l1', C=C)
        self.l1 = SelectFromModel(self.base_est)

    def fitModel(self, X, y):
        self.l1.fit(X, y)
        joblib.dump(self.l1, self.save_path)


# PCA降维
class PCAFeatureReduceDim(object):
    def __init__(self, n_dim=10, save_path='./temp/pca_model.h5'):
        self.save_path = save_path
        pca = PCA(n_components=n_dim)
        self.pca = pca


    def fitModel(self, X_data, y_data):
        X_train, X_test, y_train, y_train = train_test_split(X_data, y_data, test_size=0.2)
        self.pca.fit(X_train)
        self.n_components = self.pca.n_components_
        joblib.dump(self.pca, self.save_path)


# 训练三种特征处理模型：l1、l1l2、pca
class FeaturesSelectionModels(object):
    def __init__(self, X_data, y_data, l1_path, l1l2_path, pca_path):
        self.x_data = X_data
        self.y_data = y_data
        self.l1 = L1FeaturesSelection(save_path=l1_path)
        self.l1l2 = L1L2FeatureSelection(save_path=l1l2_path)
        self.pca = PCAFeatureReduceDim(n_dim=10, save_path=pca_path)

    # 训练不同的模型并保存
    def trainModelAndSave(self):
        X_data = self.x_data
        y_data = self.y_data
        self.l1.fitModel(X_data, y_data)
        self.l1l2.fitModel(X_data, y_data)
        self.pca.fitModel(X_data, y_data)



if __name__ == '__main__':
    # sf_ = seg_features_()
    # l1_path = sf_.l1_path
    # l1l2_path = sf_.l1l2_path
    # pca_path = sf_.pca_path
    data_frame = pd.read_excel("../DataSets/seg_model//features/xgb_excel_normed_10.xlsx")
    target = 'zLabel'
    predictors = [x for x in data_frame.columns if x not in [target]]
    X_data, y_data = data_frame[predictors], data_frame[target].values
    fs = FeaturesSelectionModels(X_data,y_data,l1_path,l1l2_path,pca_path)
    fs.trainModelAndSave()
    