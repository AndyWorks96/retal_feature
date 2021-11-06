# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/6/13
'''
    根据定位的肿瘤区域，进一步确定不同的肿瘤区域（Complete，Core和Enh）
'''
import os
import cv2
import time
import random 
import warnings
import datetime
import pandas as pd 
import numpy as np 
import xgboost as xgb
import scipy.misc
import matplotlib.pyplot as plt
from glob import glob
from skimage import io 
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from seg_data_library import SegDataLibrary
from seg_model_features import FeaturesSelectionModels
from loc_tumor import LocationTumor
from file_path import seg_tumor_, seg_data_library_, cnn_features_, seg_features_
from util import Utils
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

radius = 5
st_ = seg_tumor_(radius)
excel_scale_path = st_.excel_scale_path
excel_raw_path = st_.excel_raw_path
excel_normed_path = st_.excel_normed_path
seg_model_path = st_.seg_model_path
sdl_ = seg_data_library_(radius)
patch_scale_path = sdl_.patch_scale_path



class FeatureSelection_DE(object):
    def __init__(self):
        # data_path = 'G:/SegLGGHGGMRI/DataSets/seg_model/select_features/DE_features_bk.xlsx'    # 37维特征集合
        # data_path = './FeaturesFusion/temp/DE_features.xlsx'    # 178维特征集合
        data_path = '../DataSets/DE_result/DE_features.xlsx'
        data = pd.read_excel(data_path)
        self.columns = data.columns 


    def setDeleteColumns(self, columns):
        predictors = columns
        self.delete_columns = [x for x in predictors if x not in self.columns]
        # 
       

    def getDeletedData(self, data):
        new_data = data.drop(self.delete_columns, axis=1)
        return new_data





class XGBoostModel(object):
    def __init__(self):
        st_ = seg_tumor_(radius)
        self.excel_scale_path = st_.excel_scale_path
        self.excel_raw_path = st_.excel_raw_path
        self.excel_normed_path = st_.excel_normed_path
        self.seg_model_path = st_.seg_model_path
        self.excel_selected_path = '../temp/DE_features.xlsx'   # 特征选择后的保存结果
        # self.label_maps = {'0':0, '1':1, '2':2, '4':3}
        self.label_maps = {'0':0, '1':1, '2':2, '3':3}
        self.featureSelect_DE = FeatureSelection_DE()


        #xgb_excel_raw -> normal  scale excel 注释掉de即可运行
    def dataSuitable2XGBoost(self):
        excel_raw_path = self.excel_raw_path
        excel_normed_path = self.excel_normed_path
        excel_scale_path = self.excel_scale_path
        label_maps = self.label_maps
        print(excel_raw_path)
        data_frame = pd.read_excel(excel_raw_path)
        # data_frame = data_frame.drop(solumns, axis=1, inplace=True)
        target = 'zLabel'
        predictors = [x for x in data_frame.columns if x not in [target]]
        print(predictors)

        df_min = data_frame[predictors].min()   # 存储对应的scale
        df_max = data_frame[predictors].max()
        df_scale = pd.DataFrame([df_min, df_max])
        s_writer = pd.ExcelWriter(excel_scale_path)
        df_scale.to_excel(s_writer, index=False, encoding='utf-8')
        s_writer.save()

        df_normed = data_frame[predictors].apply(lambda x: (x-np.min(x)) / (np.max(x)-np.min(x)))
        df_normed[target] = data_frame[target]
        print(df_normed[target])
        df_normed[target] = [label_maps[str(s)] for s in df_normed[target]]
        df_normed = shuffle(df_normed)#李广注释掉了
        df_normed.fillna(0, inplace=True)
        #self.trainFeatureSelectinModel(df_normed)   # 用归一化的数据训练特征选择模型李广注释掉
        n_writer = pd.ExcelWriter(excel_normed_path)
        df_normed.to_excel(n_writer, index=False, encoding='utf-8')
        n_writer.save()


    # 进行特征选择并将选择结果进行保存
    def doFeatureSelection(self):
        excel_data = pd.read_excel(self.excel_normed_path)
        self.featureSelect_DE.setDeleteColumns(excel_data.columns)
        print(len(self.featureSelect_DE.delete_columns))
        # new_data = self.featureSelect_DE.getDeletedData(excel_data)
        # new_data.to_excel(self.excel_selected_path, index=False, encoding='utf-8')


    def loadDataSets(self):
        #excel_selected_path = self.excel_selected_path
        excel_selected_path = "D:/BraTS2019_Experiments/DataSets/seg_model/features/xgb_excel_normed_no_cnn.xlsx"
        data_frame = pd.read_excel(excel_selected_path)
        #data_frame = shuffle(data_frame)
        target = 'zLabel'
        predictors = [x for x in data_frame.columns if x not in [target]]

        X_data, y_data = data_frame[predictors], data_frame[target].values
        # train_x, test_x, train_y, test_y = train_test_split(X_data, y_data, test_size=0.2)
        # return train_x, train_y, test_x, test_y
        return X_data, y_data


    # 根据参数和估计器进行调参
    def tuneParameters(self, param_test, estimator, train_x, train_y):
        gsearch = GridSearchCV(estimator=estimator,
                               param_grid=param_test,
                               scoring='accuracy',
                               n_jobs=4,
                               iid=False,
                               cv=5)
        gsearch.fit(train_x, train_y)
        print(gsearch.best_params_, gsearch.best_score_)



    # 训练模型
    def fitXGBoost(self, alg, useTrainCV=False, cv_folds=5, eraly_stopping_rounds=50):
        train_x, train_y, test_x, test_y = self.loadDataSets()
        if useTrainCV:  
            xgb_param = alg.get_xgb_params()
            xgtrain = xgb.DMatrix(train_x, label=train_y)    # 数据使用方式
            cvresult = xgb.cv(xgb_param, xgtrain,    # 使用交叉验证
                              num_boost_round=alg.get_xgb_params()['n_estimators'], 
                              nfold=cv_folds, 
                              metrics='merror',   # 该数据集无法使用AUC进行验证
                              early_stopping_rounds=eraly_stopping_rounds)
            alg.set_params(n_estimators=cvresult.shape[0])

            alg.fit(train_x, train_y, eval_metric='auc')
            print('The best paramters is:\n', alg.get_xgb_params())

            dtrain_predictions = alg.predict(test_x)
            dtrain_predprob = alg.predict_proba(test_x)[:, 1]
            print('\n Model Report:\n')
            print('Accuracy(Train): %.4g' % metrics.accuracy_score(test_y, dtrain_predictions))

        else:
            alg.fit(train_x, train_y)
            train_result = alg.predict(train_x)
            joblib.dump(alg, seg_model_path)
            print('Model save to {}'.format(seg_model_path))
            test_predictions = alg.predict(test_x)
            print('Ground truth:')
            print(test_y)
            print('\nPredict:')
            print(test_predictions)
            print('\nModel Report:\n')
            print('Accuracy(Test): %.4g\n' % metrics.accuracy_score(test_y, test_predictions))




def trainModel():
    xgb_model = XGBoostModel()
    #xgb_model.dataSuitable2XGBoost()
    #xgb_model.doFeatureSelection() 
    train_x, train_y, test_x, test_y = xgb_model.loadDataSets()

    #param_test = {}
    xgb = XGBClassifier(learning_rate=0.1,
                         n_estimators=183,   # 127
                         max_depth=9,
                         min_child_weight=1, 
                         gamma=0.1,   # 0.5
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective='multi:softmax',
                         num_class=4,
                         nthread=4, 
                         scale_pos_weight=1,
                         seed=27)
    ##xgb_model.tuneParameters(param_test, xgb, train_x, train_y)

    xgb_model.fitXGBoost(alg=xgb, useTrainCV=False)
    # xgb_model.fitXGBoost(alg=xgb, useTrainCV=True)


if __name__ == '__main__':
    # trainModel()
    xgb_model = XGBoostModel()
    data_x, data_y = xgb_model.loadDataSets()###全部得x，y
    seg_model = joblib.load(seg_model_path)
    test_predictions = seg_model.predict(data_x)
    print('Ground truth:')
    print(data_y)
    print('\nPredict:')
    print(test_predictions)
    print('\nModel Report:\n')
    print('Accuracy(Test): %.4g\n' % metrics.accuracy_score(data_y, test_predictions))
    dets= np.savetxt('xgboost_segresult.txt',test_predictions,delimiter=',')