# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/6/14
'''
    根据原始数据生成标注模型的训练数据
    1. 主要分为三类：___、___和___
    2. 由于部分类别的肿瘤区域十分小，估将采样条件适当缩放，只要该区域80%的数据属于该类型就可作为训练集使用

    步骤：
    1. 根据每个病人的四个数据序列生成(155, 5, 240, 240)的原始矩阵
    2. 根据每个原始矩阵提取三种类别的patch(4, grid, grid)并获得对应的label
    3. 根据获得的patch分别提取四种数据序列的特征并获得标签存储为excel数据
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
from FeatureExtraction import FirstOrderStatistics, TexturalFeatures, WaveletFeatures, CNNFeatures
from file_path import seg_data_library_, cnn_features_

radius = 5
sdl_ = seg_data_library_(radius)
dcm_root = sdl_.dcm_root
raw_save_dir = sdl_.raw_save_dir
c_f_ = cnn_features_(radius=radius)
features_save = c_f_.features_save

class SegDataLibrary(object):
    def __init__(self, train_dir='', label_dir=''):
        self.train_dir = train_dir
        self.label_dir = label_dir
        self.total_patchs = 2400
        self.label_lengths = [[], [1000, 500, 100], [1000, 500, 100], [], [500, 200, 80]]

        self.fos = FirstOrderStatistics.FirstOrderStatistics()
        self.glcm = TexturalFeatures.GLCM()
        self.glrlm = TexturalFeatures.GLRLM()
        self.hog = TexturalFeatures.HOG()
        self.lbp = TexturalFeatures.LBP()

        sdl_ = seg_data_library_(radius)
        self.raw_save_dir = sdl_.raw_save_dir
        self.radius = sdl_.radius
        self.train_dir = sdl_.train_dir
        self.label_dir = sdl_.label_dir
        self.patch_train_path = sdl_.patch_train_path
        self.patch_label_path = sdl_.patch_label_path
        self.patch_scale_path = sdl_.patch_scale_path   # npy缩放模型
        self.all_patch_scale_path = '../DataSets/seg_model/xgb_all_patch_scale.npy'
        self.excel_raw_path = sdl_.excel_raw_path   # excel缩放模型
        self.all_np_scale_model = np.load(self.all_patch_scale_path)       


    # 设置特征提取器
    def setFeaturesExtractor(self, radius):
        c_f_ = cnn_features_(radius=radius)
        features_save = c_f_.features_save
        self.cnn = CNNFeatures.CNNFeatures(model_dir=features_save)

    # 将原始dcm数据处理成npy矩阵并保存（为了方便处理并且保持数据不变）
    def disposeRawDatas(self, dcm_dir, save_dir):
        print('Loading scans...')
       
        slices_by_mode = np.zeros((5, 155, 240, 240))
        slices_by_slices = np.zeros((155, 5, 240, 240))
        
        flair = glob(dcm_dir + '/*flair.dcm')
        t1 = glob(dcm_dir + '/*t1.dcm')
        t1_ce = glob(dcm_dir + '/*t1ce.dcm')
        t2 = glob(dcm_dir + '/*t2.dcm')
        seg = glob(dcm_dir + '/*seg.dcm')
        scans = [flair[0], t1[0], t1_ce[0], t2[0], seg[0]]
        patient_name = scans[0][scans[0].rfind("\\")+1: scans[0].rfind('.')]

        for scan_idx in range(5):    # 转换数据格式
            slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
        for mode_idx in range(slices_by_mode.shape[0]):
            for slice_idx in range(slices_by_mode.shape[1]):
                slices_by_slices[slice_idx][mode_idx] = slices_by_mode[mode_idx][slice_idx]   # reshape数据为CNN的输入
        print(slices_by_slices.shape)
        for s_idx in range(155):    # 根据条件进行性保存
            strips = slices_by_slices[s_idx]
            tmp = strips[-1]##segment图像
            result = np.argwhere(tmp!=0)#返回不等于0的结果如果=0那么result为[]
            if len(result)>self.total_patchs:#如果result的结果大于total那么才保存
                # pass
                np.save(save_dir + '/{}_{}.npy'.format(patient_name, s_idx), strips)
        return slices_by_slices, patient_name


    # 缩放模型
    def scaleSlices(self, trains):
        np_scale = self.all_np_scale_model
        for i in range(4):
            min_val, max_val = np_scale[i]
            if min_val != max_val:
                trains[i, :, :] -= min_val
                trains[i, :, :] *= (768.0/(max_val - min_val))
                print(train)

        return trains


    # 获得不同类别的边界（label={1, 2, 3}）并返回
    def getBoundaryCoordinates(self, array, label):
        array = array[0]
        array[array!=label] = 0
        plt.imshow(array)
        #plt.show()
        cnt = np.argwhere(array==label)
        return np.array(cnt) if np.any(array==label) else np.array([]) 

    # 获得每个病人不同类别的patch
    def getPatchsByLabel(self, images_t, images_l, radius, sentinel):
        train, label = [], []
        img_train = images_t[:4].reshape(4, 240, 240)
        img_train = self.scaleSlices(img_train)
        img_label = images_l[4:].reshape(1, 240, 240) 

        grid = 2*radius
        
        cnt = self.getBoundaryCoordinates(img_label, sentinel)
        length = len(cnt)
        num, search_times = 0, 0
        if length>self.label_lengths[sentinel][0]:
            num, search_times = 10, 300
        elif length > self.label_lengths[sentinel][1]:
            num, search_times = 5, 200
        elif length > self.label_lengths[sentinel][2]:
            num, search_times = 1, 100
        else:
            num, search_times = 0, 0
        
        for i in range(num):
            flag = True
            times = 1
            while (flag and times <= search_times):
                idx = random.randint(0, len(cnt)-1)
                x, y = cnt[idx]
                l_x, l_y = x - radius, y - radius
                r_x, r_y = x + radius, y + radius
                array = np.zeros((4, grid, grid))
                tmp_train = img_train[:, l_y:r_y, l_x:r_x]
                tmp_label = img_label[0, l_y:r_y, l_x:r_x]
                numbers = np.argwhere(tmp_label==sentinel)
                if(((float(len(numbers)/(grid*grid)))>=0.8) and (tmp_train.shape[1]==tmp_train.shape[2]==grid) and np.all(tmp_train)>=0):
                    # array[:, :, :] == tmp_train[:, :, :]
                    train.append(tmp_train)
                    label.append([sentinel])
                    flag = False
                times += 1 

        return np.array(train), np.array(label)

    # 获取所有病人的patch(包含三种类型)
    def getDiseasedPatchs(self, radius, save_patch=False):
        trains, labels = [], []
        # train_dir = self.train_dir
        # label_dir = self.label_dir
        # train_dir = '../DataSets/Brain_Pipelined/train_save/'
        label_dir = '../DataSets/Npy_Datas/'
        #label_dir = '../DataSets/test/'
        paths = [label_dir + '/' + s for s in os.listdir(label_dir)]
        random.shuffle(paths)
        sentinels = [1, 2, 4]
        for s in sentinels:
            numbers = 0
            for p in paths:
                name = p[p.rfind('/')+1: p.rfind('.')]
                #images_t = io.imread(p).reshape(5, 240, 240).astype('float')
                images_t = np.load(p).reshape(5, 240, 240).astype('float')
                print(glob(label_dir + name + '*'))
                images_l = np.load(glob(label_dir + name + '*')[0])
                train, label = self.getPatchsByLabel(images_t, images_l, radius, sentinel=s)
                trains.extend(train)
                labels.extend(label)
                numbers += len(train)
                print(s)
                if (numbers > self.total_patchs):
                    break
                
            print('The num of label {} is: {}'.format(s, numbers))
        
        if save_patch==False:
            return trains, labels
        else:
            patch_train_path = self.patch_train_path
            patch_label_path = self.patch_label_path
            if not os.path.exists(patch_train_path):
                os.makedirs(patch_train_path)
            if not os.path.exists(patch_label_path):
                os.makedirs(patch_label_path)
            np.save(patch_train_path + 'seg_train_' + str(2*radius) + '.npy', trains)
            np.save(patch_label_path + 'seg_label_' + str(2*radius) + '.npy', labels)
            return train_save_path, label_save_path



    # 正常patch的获得标准
    def normPatch(self, images_t, images_l, radius, num):
        train, label = [], []
        img_train = images_t[:4].reshape(4, 240, 240)
        img_train = self.scaleSlices(img_train)
        img_label = images_l[4:].reshape(1, 240, 240)
        grid = 2 * radius
        array = np.ndarray((4, grid, grid))
        cnt = np.argwhere(img_label[0]==0)
        for i in range(num):
            flag = True
            times = 1
            while (flag and times!=200):
                idx = random.randint(0, len(cnt)-1)
                x, y = cnt[idx]
                l_x, l_y = x - radius, y - radius  # 得到小方格的左上角和右下角坐标
                r_x, r_y = x + radius, y + radius
                tmp_train = img_train[:, l_y:r_y, l_x:r_x]
                tmp_label = img_label[0, l_y:r_y, l_x:r_x]
                numbers = np.argwhere(tmp_label!=0)
                if(((float(len(numbers)/(grid*grid)))<=0.01) and (tmp_train.shape[1]==tmp_train.shape[2]==grid) and np.all(tmp_train)>0):
                    # array[:, :, :] == tmp_train[:, :, :]
                    train.append(tmp_train)
                    label.append([0])
                    flag = False
                times += 1

        return np.array(train), np.array(label)

    # 获取正常的patch
    def getNormalPatchs(self, radius, num=1, save_patch=False):
        trains, labels = [], []
        # train_dir = self.train_dir
        # label_dir = self.label_dir
        # liguang code ####
        # train_dir = '../DataSets/Brain_Pipelined/train_save'
        # label_dir = '../DataSets/Brain_Pipelined/Npy_Datas'
        # paths = [train_dir + '/' + s for s in os.listdir(train_dir)]
        ####################
        # train_dir = '../DataSets/Brain_Pipelined/train_save/'
        label_dir = '../DataSets/Npy_Datas/'
        # label_dir = '../DataSets/test/'
        paths = [label_dir + '/' + s for s in os.listdir(label_dir)]
        random.shuffle(paths)
        numbers = 0
        for p in paths:
            name = p[p.rfind('/')+1: p.rfind('.')]
            #images_t = io.imread(p).reshape(5, 240, 240).astype('float')
            images_t = np.load(p).reshape(5, 240, 240).astype('float')
            images_l = np.load(glob(label_dir + name + '*')[0])
            train, label = self.normPatch(images_t, images_l, radius, num=num)
            trains.extend(train)
            labels.extend(label)
            numbers += len(train)
            if numbers > self.total_patchs:
                break
        print('The num of label {} is: {}'.format(0, numbers))
        if save_patch == False:
            return trains, labels
        else:
            patch_train_path = self.patch_train_path
            patch_label_path = self.patch_label_path
            if not os.path.exists(patch_train_path):
                os.makedirs(patch_train_path)
            if not os.path.exists(patch_label_path):
                os.makedirs(patch_label_path)
            np.save(patch_train_path + 'seg_train_' + str(2*radius) + '.npy', trains)
            np.save(patch_label_path + 'seg_label_' + str(2*radius) + '.npy', labels)
        



    # 提取特征(对flair、t1、t1_ce、t2分别提取特征)
    def getFeatures(self, arrays, idx=4):
        all_features = {}
        #cnn_f = self.cnn.getFeatures(arrays.reshape(1, 4, 2*radius, 2*radius))[0]
        # self.setFeaturesExtractor(4)
        cnn_f = self.cnn.getFeatures(arrays)[0]
        key = ['cnnf_' + str(s) for s in range(1, len(cnn_f)+1)]
        cnn_fs = dict(zip(key, cnn_f))

        shuffix = ['f', 't1', 't1c', 't2']
        for i in range(idx):
            array = arrays[i]
            fos_f = self.fos.getFirstOrderStatistics(array)
            glcm_f = self.glcm.getGLCMFeatures(array.astype(np.int64))
            glrlm_f = self.glrlm.getGLRLMFeatures(array.astype(np.int64))
            hog_f = self.hog.getHOGFeatures(array)
            lbp_f = self.hog.getHOGFeatures(self.lbp.getLBPFeatures(array), name='lbp')
            features = {**fos_f, **glcm_f, **glrlm_f, **hog_f, **lbp_f}
            tmp_features = {}
            for key in features:
                new_key = key + '_' + shuffix[i]
                tmp_features[new_key] = features[key]
            all_features = {**all_features, ** tmp_features}
        return {**all_features, **cnn_fs}


    # 根据获取的patch提取特征
    def getFeaturesByPatchs(self, train_save, label_save):
        self.cnn = CNNFeatures.CNNFeatures(model_dir=features_save)
        print('Starts get features...')
        patch_scale_path = self.patch_scale_path
        excel_raw_path = self.excel_raw_path
        if (type(train_save) == str):
            trains = np.load(train_save)
            labels = np.load(label_save)
        else:
            trains = train_save
            labels = label_save
        print(trains.shape)
        print(labels.shape)

        scale_model = []     # 数据归一化，并保存对应通道的结果
        for i in range(4):
             min_val = trains[:, i, :, :].min()
             max_val = trains[:, i, :, :].max()
             scale_model.append([min_val, max_val])
             if min_val != max_val:
                 trains[:, i, :, :] -= min_val
                 trains[:, i, :, :] *= (600.0/(max_val - min_val))
        scale_model = np.array(scale_model)
        print(scale_model)
        np.save(patch_scale_path, scale_model)

        data_frame = None    # 特征提取
        for i in range(len(trains)):
            print('The number {}/{}'.format(i, len(trains)))
            train = trains[i]
            label = labels[i]
            features = self.getFeatures(train, len(train))
            features['zLabel'] = label[0]
            if data_frame is None:
                data_frame = pd.DataFrame(features, index=[0])
            else:
                data_frame.loc[i] = features
        print(len(data_frame.columns), '\n')
        writer = pd.ExcelWriter(excel_raw_path)
        data_frame.to_excel(writer, index=False, encoding='utf-8')
        writer.save()
        print(data_frame.tail())
        print('Save to -->{}<-- successed!\n'.format(excel_raw_path))
        
# ----------------------------------------------------------------------------------------


# 处理原始数据
def disposeRawdatas():
    print('Start dispose raw datas...')
   
    dcm_dir = [dcm_root + '/' + s for s in os.listdir(dcm_root) ]
    random.shuffle(dcm_dir)

    for d in dcm_dir:
        seg_data_library = SegDataLibrary()
        seg_data_library.disposeRawDatas(d, raw_save_dir)

    print('Dispose raw datas finished!!!')


# 使用本次数据训练CNN特征提取模型
def trainCNNFeatreus(train_path, label_path):
    grid = 2*radius
    cnn_model = CNNFeatures.CNNModelTrains(features_save=features_save, 
                                           img_size=[4, grid, grid],
                                           n_filters=16,
                                           n_features=128
                                           )
    #batch_size=32, epochs=30
    cnn_model.fitModel(train_path, label_path)


# 提取patch并获得对应的features
def getFeaturesFromPatchs():
    radius = 5
    trains, labels = [], []
    seg_data_library = SegDataLibrary()
    patch_train_path = seg_data_library.patch_train_path
    patch_label_path = seg_data_library.patch_label_path
    train, label = seg_data_library.getDiseasedPatchs(radius=radius)
    trains.extend(train)
    labels.extend(label)
    train, label = seg_data_library.getNormalPatchs(radius=radius)
    trains.extend(train)
    labels.extend(label)
    trains = np.array(trains)
    labels = np.array(labels)
    print(trains.shape)
    print(labels.shape)
    print(trains[1].shape, trains[1])
    np.save(patch_train_path, trains)
    np.save(patch_label_path, labels)

    trainCNNFeatreus(patch_train_path, patch_label_path)

    seg_data_library.getFeaturesByPatchs(train_save=patch_train_path, label_save=patch_label_path)


if __name__ == '__main__':
    #disposeRawdatas()
    getFeaturesFromPatchs()
    #####检测图形形状###
    # images_test = np.load("../DataSets/Raw_datas/BraTS19_2013_11_1_flair_0.npy")
    # img = cv2.imread("../DataSets/Brain_Pipelined/BraTS19_2013_11_1_flair_47.png")
    # print(images_test.shape)
    # print(img.shape)
    ##################
    # train_dir = './Brain_Pipelined/'
    # label_dir = './DataSets/Raw_Datas/'
    # trains_paths = [train_dir + '/' +s for s in os.listdir(train_dir)]
    # for i in range(len(trains_paths)):
    #      path = trains_paths[i]
    #      name = path[path.rfind('/')+1: path.rfind('.')]
    #      train = io.imread(path)
    #      label = np.load(glob(label_dir + name + '*')[0]).reshape(1200, 240)
    #      plt.imshow(train)
    #      plt.show()
    #      plt.imshow(label)
    #      plt.show()
