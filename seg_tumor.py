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
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from seg_data_library import SegDataLibrary
from loc_tumor import LocationTumor
from file_path import seg_tumor_, seg_data_library_
from util import Utils
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

radius = 12
st_ = seg_tumor_(radius)
excel_scale_path = st_.excel_scale_path
excel_raw_path = st_.excel_raw_path
excel_normed_path = st_.excel_normed_path
seg_model_path = st_.seg_model_path
sdl_ = seg_data_library_(radius)
patch_scale_path = sdl_.patch_scale_path


class SegmentationTumor(object):
    def __init__(self, seg_model_path='', excel_scale_path='', patch_scale_path='', seg_save_path=''):
        sdl_ = seg_data_library_(radius)
        st_ = seg_tumor_(radius)
        self.patch_scale_path = sdl_.patch_scale_path
        self.all_patch_scale_path = '../DataSets/xgb_all_patch_scale.npy'
        self.excel_scale_path = st_.excel_scale_path
        self.seg_model_path = st_.seg_model_path
        self.seg_model_path = '../DataSets/seg_model/ModelWeights/xgb_seg_model_10.h5'
        self.seg_save_path = seg_save_path
        
        self.seg_model = joblib.load(self.seg_model_path)
        self.df_scale = pd.read_excel(self.excel_scale_path)
        self.np_scale = np.load(self.patch_scale_path)
        self.segDataLibrary = SegDataLibrary()
        print('patch_scale_path: {}'.format(self.patch_scale_path))
        print('excel_scale_path: {}'.format(self.excel_scale_path))
        print('seg_model_path: {}'.format(self.seg_model_path))


    # 测试时提取特征的方法
    def getFeatures(self, array):
        df_scale = self.df_scale
        segDataLibrary = self.segDataLibrary
        np_scale = self.np_scale
        segDataLibrary = self.segDataLibrary
        for i in range(4):
            min_val, max_val = np_scale[i]
            if min_val != max_val:
                array[i, :, :] -= min_val
                array[i, :, :] *= (600.0/(max_val - min_val))
        features = segDataLibrary.getFeatures(array)
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
            return predict[0]
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
    def segTumor(self, dcm_path, locations=[], loc_path='', grid=10, radius=5, seg_name=''):
        row, col = 240, 240
        npy_array = io.imread(dcm_path).reshape(5, 240, 240).astype('float')
        dcm_array = npy_array[:4].reshape(4, 240, 240)
        gray_dcm_path = './tmp_dcm_gray.jpg'    # ---------------------------保存灰度图像
        scipy.misc.toimage(dcm_array.reshape(960, 240), cmin=0, cmax=255).save(gray_dcm_path)
        # for i in range(4):
        #     min_val, max_val = self.np_scale[i]
        #     if min_val != max_val:
        #         dcm_array[i, :, :] -= min_val
        #         dcm_array[i, :, :] *= (255.0/(max_val - min_val))
        label = npy_array[-1].reshape(240, 240)
        
        starttime = datetime.datetime.now()    # ----------------------------------开始时间

        ks = 1
        searched_locs = []
        array = np.ndarray((4, grid, grid))
        mask = np.zeros((row, col))
        for loc in locations:
            x, y = loc[0], loc[1]
            print('Processing loc {}/{} ...'.format(ks, len(locations)))
            direction = [(0, 1), (-1, 1), (-1, 0), (-1, -1),
                         (0, -1), (1, -1), (1, 0), (1, 1)]
            for dix in direction:
                x_d, y_d = dix
                # print(x_d, y_d)
                k = 0
                flag = True
                x1, y1 = radius+1, radius+1
                # while(flag and radius<x1<row and radius<y1<col):
                while(k!=30):
                    x1, y1 = x + k*x_d, y + k*y_d
                    # print(x1, y1)
                    if ([x1, y1] not in searched_locs):
                        tmp = dcm_array[:, y1-radius: y1+radius, x1-radius: x1+radius]
                        # print(tmp)
                        if tmp.shape[1] == tmp.shape[2] == grid:
                            label = self.getPixelLabel(tmp, radius)
                            mask[y1, x1] = label
                            print(label)
                            # if (label == 0 or label == 1) and np.all(loc_Result[y1, x1, :] != [0, 0, 255]):
                            if label == 0 :
                                flag = False
                    k += 1
                    searched_locs.append([x1, y1])
            ks += 1
 
        print("The SegTumor Model time: {}s".format((datetime.datetime.now() - starttime).seconds))
        plt.imshow(mask)
        plt.show()
        # self.plotSegResult(mask, label, gray_dcm_path, loc_path, seg_name)




def segTumor(dcm_path):
    # dcm_path = './DataSets/Brain_Pipelined/Brats17_TCIA_408_1_flair_52.png'
    locTumor = LocationTumor()
    scale_model_path = '../DataSets/loc_model/ModelWeights/loc_patch_scale_' + str(26) + '.npy'
    # loc_model_path = './ModelWeights/norm_tumor_Model/loc_model_nt_' + str(26) + '.h5'
    loc_model_path = '../DataSets/loc_model/ModelWeights/loc_ord_model_26.h5'
    locations = locTumor.locationTumor(loc_model_path, scale_model_path, img_path=dcm_path, grid=26, radius=13, step=13)
    
    
    print(seg_model_path)
    segmentation_tumor = SegmentationTumor(seg_model_path, excel_scale_path, patch_scale_path)
    loc_path = ''
    seg_name = './temp/this_is_one_test.jpg'
    segmentation_tumor.segTumor(dcm_path=dcm_path, locations=locations, loc_path=loc_path, seg_name=seg_name)

    utils = Utils()
    label_img = np.load(Utils.matchPathWithLabels('./DataSets/Raw_Datas/', dcm_path))[-1]
    print(label_img.shape)
    plt.imshow(label_img)
    plt.show()

if __name__ == '__main__':
    dcm_paths = ['../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_47.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_48.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_49.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_50.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_51.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_52.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_53.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_54.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_55.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_56.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_57.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_58.png',
                 '../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_59.png']
    for dcm_path in dcm_paths:
        segTumor(dcm_path)
    # segTumor()

