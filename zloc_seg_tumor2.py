# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/6/27
'''
    集成的最终分类模型，包括疑似区域定位和肿瘤标注模型
'''
import os
import cv2
import time
import keras
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
from keras.models import *
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from seg_data_library import SegDataLibrary
from file_path import loc_tumor_, seg_data_library_, seg_tumor_, cnn_features_, seg_features_
from util import Utils
from evalution_index import EvalutionIndex, SaveEvaltuion

class FeatureSelection_DE(object):
    def __init__(self):
        #data_path = '../DataSets/seg_model/select_features/DE_features.xlsx'    # 37个特征集合
        data_path = '../DataSets/DE_result/DE_features.xlsx'
        data = pd.read_excel(data_path)
        self.columns = data.columns 


    def setDeleteColumns(self):
        raw_data_path = '../DataSets/seg_model/features/xgb_excel_normed_10.xlsx'   # 未经过特征选择时的特征集合
        raw_data = pd.read_excel(raw_data_path)
        predictors = raw_data.columns
        self.delete_columns = [x for x in predictors if x not in self.columns]

class LabelTumor(object):
    def __init__(self, loc_radius=0, seg_radius=[0, 0]):
        lt_ = loc_tumor_(loc_radius)
        self.loc_radius = loc_radius
        self.loc_grid = 2*loc_radius
        self.loc_step = loc_radius
        self.loc_patch_scale_path = lt_.loc_patch_scale_path
        self.loc_model_path = lt_.loc_model_path
        # self.loc_patch_scale_path = './DataSets/loc_model/ModelWeights/loc_patch_scale_hgg_' + str(26) + '.npy'
        # self.loc_model_path  = './DataSets/loc_model/ModelWeights/loc_ord_model_hgg_26.h5'

        self.seg_radius_train = seg_radius[0]
        self.seg_radius_valid = seg_radius[1]
        self.seg_grid_train = 2*self.seg_radius_train
        self.seg_grid_valid = 2*self.seg_radius_valid
        st_t = seg_tumor_(self.seg_radius_train)
        self.seg_model_path = st_t.seg_model_path
        sdl_v = seg_data_library_(self.seg_radius_valid)
        st_v = seg_tumor_(self.seg_radius_valid)
        self.seg_patch_scale_path = sdl_v.patch_scale_path
        self.seg_excel_scale_path = st_v.excel_scale_path
        
        self.seg_model = joblib.load(self.seg_model_path)
        self.seg_df_scale = pd.read_excel(self.seg_excel_scale_path)
        # self.seg_np_scale = np.load(self.seg_patch_scale_path)
        self.segDataLibrary = SegDataLibrary()
        self.segDataLibrary.setFeaturesExtractor(seg_radius[0])
        print('patch_scale_path: {}'.format(self.seg_patch_scale_path))
        print('excel_scale_path: {}'.format(self.seg_excel_scale_path))
        print('seg_model_path: {}'.format(self.seg_model_path))
        self.all_patch_scale = np.load('../DataSets/seg_model/xgb_all_patch_scale.npy')


        self.feature_selection_DE = FeatureSelection_DE()
        self.feature_selection_DE.setDeleteColumns()
        # s_f_ = seg_features_(radius=self.seg_radius_train)
        # l1_path = s_f_.l1_path
        # l1l2_path = s_f_.l1l2_path
        # pca_path = s_f_.pca_path
        # self.select_model = joblib.load(l1l2_path)    # 特征选择
        # # self.reduce_model = joblib.load(pca_path)


    # 肿瘤定位模型，返回疑似区域边界
    def tumorLocation(self, img_path):
        step = self.loc_step
        radius = self.loc_radius
        grid = self.loc_grid
        loc_model = load_model(self.loc_model_path)
        scale_model = np.load(self.loc_patch_scale_path)

        img_name = img_path[img_path.rfind('/'): img_path.rfind('.')]
        image = io.imread(img_path).reshape(5, 240, 240).astype('float')
        img = image[:4].reshape(4, 240, 240)     # 待处理的图像
        for i in range(4):
            min_val, max_val = scale_model[i]
            img[i, :, :] -= min_val
            img[i, :, :] *= (1.0/(max_val - min_val))
        label = image[4:].reshape(240, 240)
        row = col = 240

        starttime = datetime.datetime.now()    

        mask = np.zeros((row, col))        # 第一次粗略搜索
        for x in range(radius, col-radius, step):   
            for y in range(radius, row-radius, step):
                l_x, l_y = x - radius, y - radius    # 得到小方格的左上角和右下角坐标
                r_x, r_y = x + radius, y + radius
                array = np.ndarray((4, grid, grid))    # 获取方格内的信息
                array[:, :, :] = img[:, l_y:r_y, l_x:r_x]
                if np.all(array>0):
                    array = array.reshape((1, array.shape[0], array.shape[1], array.shape[2]))
                    f = loc_model.predict(array)
                    result = np.argmax(f)
                    # print(result)
                    mask[l_y:r_y, l_x:r_x] = result + 1
                else:
                    mask[y, x] = 0

        mask_path = '../temp/temp_loc_mask.jpg'
        # scipy.misc.toimage(mask).save(mask_path)
        #Image.fromarray(mask).convert("L").save(mask_path)
        io.imsave(mask_path,mask)
        print('The location Model time: {}s'.format(datetime.datetime.now() - starttime))

        img_mask = cv2.imread(mask_path)        # ---------------------------寻找最大轮廓边界
        imgray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 200, 255, 1)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_sorted = sorted(contours, key=lambda x : len(x))
        # cnt = contours_sorted[-2]
        lengths = len(contours_sorted)
        if lengths >= 2:
            cnt = contours_sorted[-2]
        elif lengths > 0:
            cnt = contours_sorted[-1]
        else:
            cnt = [[0, 0]]
        img_mask = cv2.drawContours(img[1, :, :], [cnt], 0, (0, 2, 255), 2)
        show_contours = np.zeros((row, col))  
        for m in cnt:
            x,y = m[0]
            show_contours[x,y] = 1
        # plt.imshow(show_contours)
        # plt.show()
        show_img = img[1, :, :]
        show_img[mask==2] = 1
        # plt.imshow(show_img)
        # plt.show()
        
        return np.array(cnt).squeeze()


    # 根据疑似区域边界，逐个搜索像素判断像素类别
    def tumorSegmentation(self, img_path, locations):
        row, col = 240, 240
        grid = self.seg_grid_valid
        radius = self.seg_radius_valid
        npy_array = io.imread(img_path).reshape(5, 240, 240).astype('float')
        dcm_array = npy_array[:4].reshape(4, 240, 240)
        np_scale = self.all_patch_scale
        for i in range(4):
            min_val, max_val = np_scale[i]
            dcm_array[i, :, :] -= min_val
            dcm_array[i, :, :] *= (768.0/(max_val - min_val))
        gray_dcm_path = '../temp/tmp_dcm_gray.jpg'    # ---------------------------保存灰度图像
        #scipy.misc.toimage(dcm_array.reshape(960, 240), cmin=0, cmax=255).save(gray_dcm_path)
        io.imsave(gray_dcm_path,dcm_array.reshape(960, 240))
        label = npy_array[-1].reshape(240, 240)

        starttime = datetime.datetime.now()    # ----------------------------------开始时间

        ks = 1
        searched_locs = []
        array = np.ndarray((4, grid, grid))
        mask = np.zeros((row, col))
        for loc in locations:
            #print("loc={},locations={}".format(loc,locations))
            x, y = loc[0], loc[1]
            print('Processing loc {}/{} ...'.format(ks, len(locations)))
            direction = [(0, 1), (-1, 1), (-1, 0), (-1, -1),
                          (0, -1), (1, -1), (1, 0), (1, 1)]
            
            for dix in direction:
                start = time.clock()
                print("dix={},direction={}".format(dix,direction))
                x_d, y_d = dix
                # print(x_d, y_d)
                k = 0
                flag = True
                x1, y1 = radius+1, radius+1
                while(flag and radius<x1<row and radius<y1<col):
                # while(k!=30):
                    x1, y1 = x + k*x_d, y + k*y_d
                    print(x1, y1)
                    if ([x1, y1] not in searched_locs):
                        tmp = dcm_array[:, y1-radius: y1+radius, x1-radius: x1+radius]
                        if tmp.shape[1] == tmp.shape[2] == grid:
                            label = self.getPixelLabel(tmp, radius)
                            #print("label?",label)
                            mask[y1, x1] = label
                            # print(label)
                            # if (label == 0 or label == 1) and np.all(loc_Result[y1, x1, :] != [0, 0, 255]):
                            if label == 0:
                                flag = False
                    k += 1
                    searched_locs.append([x1, y1])
                elapsed = (time.clock() - start)
                print("Time used:",elapsed)
            ks += 1
 
        print("The SegTumor Model time: {}s".format((datetime.datetime.now() - starttime).seconds))
        plt.imshow(mask)
        plt.show()
        mask[mask==3] = 4
        return mask

    # 判断像素类别
    def getPixelLabel(self, array, radius):
        cmin, cmax = 0, 600
        grid = 2*radius
        prediction = 0
        flag = True
        if np.any(array<cmin):
            return predict
        # elif np.any(array==cmin):
        #     total = np.argwhere(array==cmin)
        #     if (float(len(total)/(grid*grid))) <0.8:
        #         flag = False
        # else:
        #     pass

        if flag:
            features = self.getFeatures(array)
            prediction = self.seg_model.predict(features)
            return prediction[0]
        return prediction 


    # 获取特征
    def getFeatures(self, array):
        df_scale = self.seg_df_scale
        #df_scale = self.seg_df_norm
        # np_scale = self.seg_np_scale
        # for i in range(4):
        #     min_val, max_val = np_scale[i]
        #     if min_val != max_val:
        #         array[i, :, :] -= min_val
        #         array[i, :, :] *= (600.0/(max_val - min_val))
        features = self.segDataLibrary.getFeatures(array)
        data_frame = pd.DataFrame([features])
        df_normed = (data_frame - df_scale.min()) / (df_scale.max() - df_scale.min())
        df_normed.fillna(0, inplace=True)
        df_normed[np.isinf(df_normed)] = 1

        # new_X_data = self.select_model.transform(df_normed.values)
        new_X_data = df_normed.drop(self.feature_selection_DE.delete_columns, axis=1)
        return new_X_data
        #return df_normed 

def validation(img_paths):
    loc_radius = 13
    seg_radius = [5, 5]
    utils = Utils()
    seg_grid = 2*seg_radius[1]
    label_tumor = LabelTumor(loc_radius, seg_radius)
    se = SaveEvaltuion()
    total_dices = []
    i = 1
    length = len(img_paths)
    searched = []

    for k in range(2000):
        img_path = img_paths[random.randint(0, length-1)]
        if img_path in searched:
            continue
        searched.append(img_path)
        locations = label_tumor.tumorLocation(img_path)
        segmentation_tumor = label_tumor.tumorSegmentation(img_path, locations)
        label_img = np.load(Utils.matchPathWithLabels('../DataSets/Brain_Pipelined/numpy/', img_path)).reshape(1,240,240)
        name = img_path[img_path.rfind('/')+1: img_path.rfind('.')] + '_' + str(seg_grid)
        segmentation_tumor = segmentation_tumor.reshape(1,240,240)
        print(segmentation_tumor.shape, label_img.shape)
        #plt.imshow(segmentation_tumor)
        #plt.imshow(label_img)
        #plt.show() 


        seg_array = np.concatenate((label_img, segmentation_tumor), axis=1)
        np.save('../Runed_Result/new_result/array_fig/' + name + '.npy', seg_array)
        # np.save('./temp/' + name + '.npy', seg_array)

        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        axs[0].imshow(label_img.reshape(240,240))
        axs[0].set_title('Ground truth')
        axs[1].imshow(segmentation_tumor.reshape(240,240))
        axs[1].set_title('Seg result')
        fig.suptitle(name, fontsize=14)
        plt.savefig('../Runed_Result/new_result/jpg_fig/' + name + '.jpg')
        # plt.savefig('./temp/' + name + '.jpg')
        evaluate_score = calcuteEvaluteIndex(label_img, segmentation_tumor)
        print('The img >{}<:'.format(name))
        print('\t 1 dice: {}\n\t 2 sensitivity: {}\n\t 3 spensifity: {}\n'.format(evaluate_score[0], evaluate_score[1], evaluate_score[2]))
    
        se.addSeries(name, evaluate_score)
        # if (k % 3 == 0):
        #     se.save('./Runed_Result/Seg_figure/seg_result.csv')
        se.save('../Runed_Result/new_result/seg_result8.csv')

def calcuteEvaluteIndex(ground_truth, segmentation):
    ei = EvalutionIndex()
    dice_score = []
    sen_score = []
    spe_score = []
    for label_idx in [1, 2, 4]:
        # print(ground_truth.reshape((240, 240)))
        if np.any(ground_truth==label_idx):
            dice = ei.getDiceCoefficent(ground_truth, segmentation, label_idx)
            sen = ei.getSensitivity(ground_truth, segmentation, label_idx)
            spe = ei.getSpecificity(ground_truth, segmentation, label_idx)
        else:
            dice = -1
            sen = -1 
            spe = -1
        dice_score.append(dice)
        sen_score.append(sen)
        spe_score.append(spe)
    # sen_score.extend(spe_score)
    # dice_score.extend(sen_score)
    return [dice_score, sen_score, spe_score]


if __name__ == '__main__':
    # xxx = "D:/BraTS2019_Experiments/DataSets/Brain_Pipelined/train_save_np/BraTS19_CBICA_BFB_1_flair_58.npy"
    # hj = np.load(xxx).reshape(1200,240)
    # plt.imshow(hj)
    # plt.show()
    data_dir = '../DataSets/Brain_Pipelined/train_save/'
    dcm_paths = [data_dir + '/' + s for s in os.listdir(data_dir)]
    validation(dcm_paths)


    # img_path = './DataSets/Brain_Pipelined/Brats17_2013_7_1_flair_111.png'
    # loc_radius = 13
    # seg_radius = [4, 4]
    # utils = Utils()
    # seg_grid = 2*seg_radius[1]
    # label_tumor = LabelTumor(loc_radius, seg_radius)
    # locations = label_tumor.tumorLocation(img_path)



