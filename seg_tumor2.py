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
from PIL import Image
from glob import glob
from skimage import io 
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
from keras.layers import merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Softmax, concatenate, Dense, Flatten
from keras.models import *
from keras.optimizers import *
from keras.regularizers import L1L2
from keras.utils import to_categorical, plot_model
from seg_data_library import SegDataLibrary
from loc_tumor import LocationTumor
from file_path import seg_tumor_, seg_data_library_
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

st_ = seg_tumor_(5)#13
excel_scale_path = st_.excel_scale_path
excel_raw_path = st_.excel_raw_path
excel_normed_path = st_.excel_normed_path
# loc_model_path = st_.loc_model_path
seg_model_path = st_.seg_model_path
sdl_ = seg_data_library_(5)#13
patch_scale_path = sdl_.patch_scale_path


class SegmentationTumor(object):
    def __init__(self, seg_model_path, xlsx_scale_path, npy_scale_path, seg_save_path=''):
        self.seg_save_path = seg_save_path
        self.seg_model_path = seg_model_path
        self.seg_model = joblib.load(self.seg_model_path)
        self.xlsx_scale_path = xlsx_scale_path
        self.df_scale = pd.read_excel(self.xlsx_scale_path)
        self.npy_scale_path = npy_scale_path
        self.np_scale = np.load(self.npy_scale_path)
        self.seg_data_library = SegDataLibrary()


    # 测试时提取特征的方法
    def getFeatures(self, array):
        df_scale = self.df_scale
        seg_data_library = self.seg_data_library
        features = seg_data_library.getFeatures(array)
        data_frame = pd.DataFrame([features])
        df_normed = (data_frame - df_scale.min()) / (df_scale.max() - df_scale.min())
        df_normed.fillna(0, inplace=True)
        return df_normed.values



    # array.shape=[4, 240, 240]
    def getPixelLabel(self, array, radius, cmin=0, cmax=255):
        grid = 2*radius
        predict = 0
        flag = True
        if np.any(array<cmin):
            return predict
        elif np.any(array==cmin):
            total = np.argwhere(array==cmin)
            if (float(len(total)/(grid*grid))) <0.8:
                flag = False
        else:
            pass

        if flag:
            features = self.getFeatures(array)
            predict = self.seg_model.predict(features)
            # print(predict)
            # print(predict + 1, '\n')
            return predict[0] + 1 
        return predict 



    # 保存定位、分割的结果
    def plotSegResult(self, mask, label, gray_dcm_path, loc_path, seg_path=''):
        segResult = np.array(io.imread(gray_dcm_path)).reshape(4, 240, 240)[-1]   # 分割结果
        print(segResult.shape)
        xn, yn = np.where(mask==2)   # 1
        xb, yb = np.where(mask==3)   # 2
        xt, yt = np.where(mask==4)   # 3
        for i in range(len(xt)):
            segResult[xt[i], yt[i], :] = [255, 0, 0]
        for i in range(len(xb)):
            segResult[xb[i], yb[i], :] = [0, 255, 0]
        for i in range(len(xn)):
            segResult[xn[i], yn[i], :] = [0, 0, 255]

        plt.imshow(segResult)
        plt.show()

        # seg_label = np.array((3, 240, 240))    # 真实结果
        # seg_label[0, :, :] = label[:, :]
        # seg_label[1, :, :] = label[:, :]
        # seg_label[2, :, :] = label[:, :]
        # x1, y1 = np.where(label==1)   # 1
        # x2, y2 = np.where(label==2)   # 2
        # x3, y3 = np.where(label==3)   # 3
        # for i in range(len(x1)):
        #     seg_label[x1[i], y1[i], :] = [255, 0, 0]
        # for i in range(len(x2)):
        #     seg_label[x2[i], y2[i], :] = [0, 255, 0]
        # for i in range(len(x3)):
        #     seg_label[x3[i], y3[i], :] = [0, 0, 255]

        # name = save_path[save_path.rfind('/')+1: save_path.rfind('_')] + '_label.png'
        # orig_img = np.array(cv2.imread(gray_dcm_path))   # [960, 240]
        # loc_img = np.array(cv2.imread(loc_path))   # [240, 240]
        # label[label!=0] = 255
        # loc_label_img = np.array(label)   # [240, 240]
        # seg_label_img = seg_label  
        # result = np.concatenate((loc_label_img, seg_label_img, loc_img, segResult), axis=1) 
        # label_img = np.concatenate((orig_img, result), axis=1)
        # plt.imshow(label_img)
        # plt.show()



    # 根据定位的结果进行进行精细分割
    def segTumor(self, dcm_path, loc_path='', grid=8, radius=8, seg_name=''):

        # dcm_array = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))
        row, col = 240, 240
        # npy_array = np.load(dcm_path).reshape(5, 240, 240).astype('float')
        npy_array = io.imread(dcm_path).reshape(5, 240, 240).astype('float')
        dcm_array = npy_array[:4].reshape(4, 240, 240)
        print(dcm_array.shape)
        gray_dcm_path = './tmp_dcm_gray.jpg'    # ---------------------------保存灰度图像
        #scipy.misc.toimage(dcm_array.reshape(960, 240), cmin=0, cmax=255).save(gray_dcm_path)
        Image.fromarray(dcm_array.reshape(960, 240)).convert("L").save(gray_dcm_path)
        #io.imsave(gray_dcm_path,dcm_array.reshape(960, 240)) 
        for i in range(4):
            min_val, max_val = self.np_scale[i]
            if min_val != max_val:
                dcm_array[i, :, :] -= min_val
                dcm_array[i, :, :] *= (255.0/(max_val - min_val))
        label = npy_array[-1].reshape(240, 240)

        starttime = datetime.datetime.now()    # ----------------------------------开始时间

        ks = 1
        searched_locs = []
        prints = [0, 0, 0, 0, 0, 0]
        array = np.ndarray((4, grid, grid))
        mask = np.zeros((row, col))
        for x in range(grid, row):
            for y in range(grid, col):
                x1, y1 = x-radius, y-radius
                x2, y2 = x+radius, y+radius
                tmp = dcm_array[:, y1:y2, x1:x2]
                if tmp.shape[1] == tmp.shape[2] == grid:
                    label = self.getPixelLabel(tmp, radius)
                    mask[y1, x1] = label
                    # print(label)
                    prints[label] += 1
        print(prints)
 
        print("The SegTumor Model time: {}s".format((datetime.datetime.now() - starttime).seconds))
        self.plotSegResult(mask, label, gray_dcm_path, loc_path, seg_name)


def segTumor():
    dcm_path = '../DataSets/Brain_Pipelined/train_save/BraTS19_CBICA_AAP_1_flair_100.png'
    locTumor = LocationTumor()
    # scale_model_path = '../DataSets/ModelWeights/norm_tumor_Model/scale_model_nt_' + str(26) + '.npy'
    # loc_model_path = '../DataSets/ModelWeights/norm_tumor_Model/loc_model_nt_' + str(26) + '.h5'
    scale_model_path = '../DataSets/loc_model/ModelWeights/loc_patch_scale_' + str(26) + '.npy'
    loc_model_path = '../DataSets/loc_model/ModelWeights/loc_ord_model_26.h5'
    locations = locTumor.locationTumor(loc_model_path, scale_model_path, img_path=dcm_path, grid=26, radius=13, step=13)
    
    
    print(seg_model_path)
    segmentation_tumor = SegmentationTumor(seg_model_path, xlsx_scale_path=excel_scale_path, npy_scale_path=patch_scale_path)
    loc_path = ''
    seg_name = './temp/this_is_one_test.jpg'
    segmentation_tumor.segTumor(dcm_path=dcm_path, loc_path=loc_path, seg_name=seg_name)


if __name__ == '__main__':
    segTumor()

    # data_frame = pd.read_excel(normed_path)
    # print(data_frame.head())