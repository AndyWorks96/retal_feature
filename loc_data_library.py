# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/5/17
'''
    根据预处理后的数据生成对应的标注数据：
    1，getBoundaryPatchs：获取标签的边界信息，从中随机选取num个数据生成patch
    2，getTumorPatchs：获取标签的内部数据（全部在标签内部），从中随机选取num个数据生成patch
    3，getNormalPatchs：获取标签外部的数据（没有落在标签内部且不包含0），从中随机选取num个生成patch
'''
import os
import cv2
import random
import numpy as np 
import matplotlib.pyplot as plt
from glob import glob
from skimage import io 
from scipy import misc
from PIL import Image

from file_path import loc_data_library_
ldl = loc_data_library_()
brain_pielined_dir = ldl.brain_pielined_dir

class LocDataLibrary(object):
    def __init__(self, paths):
        self.paths = paths

    #　获取肿瘤边界坐标
    def getBoundaryCoordinates(self, array):
        cv_tmp_jpg = './cv_temp.jpg'
        io.imsave(cv_tmp_jpg, array[0])
        #misc.toimage(array[0], cmin=0, cmax=255).save(cv_tmp_jpg)
        #Image.fromarray(array[0]).convert('RGB').save(cv_tmp_jpg)
        image = cv2.imread(cv_tmp_jpg)
        # plt.imshow(image)
        # plt.show()
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 1)
        _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
        contours_sorted = sorted(contours, key=lambda x : len(x))
        cnt = contours_sorted[-1]
        # cv2.drawContours(image, [cnt], 0, (0,0,255), 2)
        # cv2.imshow('src', image)
        # cv2.waitKey()
        return np.array(cnt).squeeze()


    def loadImages(self, train_save, label_save, types='', radius=10, total=500):
        #radius = 13 total == 2000
        paths = self.paths
        trains = []   # 训练数据
        labels = []   # 标签
        print("current type is {}".format(types))
        func_map = {'n': self.getNormalPatchs, 'b':self.getBoundaryPatchs, 't': self.getTumorPatchs}
        for i in range(len(paths)):
            img = io.imread(paths[i]).reshape(5, 240, 240).astype('float')#读取的图片是包含四个序列加groundtruth的
            img[-1][img[-1] != 0] = 1
            print("times:{}/{}, type:{} ".format(i,len(paths),types))
            # train, label = func_map[types](img, radius, int(total/len(paths))+1)
            train, label = func_map[types](img, radius, 2)
            trains.extend(train)
            labels.extend(label)
            if(len(trains)>2000):
                break

        trains = np.array(trains)
        labels = np.array(labels)
        print('The number of class-\'{}\':'.format(types))
        print('   Trains shape: {}\n   Labels shape: {}\n'.format(trains.shape, labels.shape))
        np.save(train_save, trains)
        np.save(label_save, labels)


    def getBoundaryPatchs(self, images, radius, num):
        train, label = [], []
        img_train = images[:4].reshape(4, 240, 240)
        img_label = images[4:].reshape(1, 240, 240)

        grid = 2 * radius
        array = np.ndarray((4, grid, grid))  # 获取方格内的信息
        cnt = self.getBoundaryCoordinates(img_label)
        for i in range(num):
            flag = True
            times = 1
            while (flag and times!=100):
                idx = random.randint(0, len(cnt)-1)
                x, y = cnt[idx]
                l_x, l_y = x - radius, y - radius  # 得到小方格的左上角和右下角坐标
                r_x, r_y = x + radius, y + radius
                # print('Left_P: ', (l_x, l_y))
                # print('Center_P: ', (x, y))
                # print('Right_P: ', (r_x, r_y))
                tmpt = img_train[1, l_y:r_y, l_x:r_x]
                if np.all(tmpt!=0):
                    array[:, :, :] = img_train[:, l_y:r_y, l_x:r_x]
                    train.append(array)
                    label.append([1])
                    flag = False
                times += 1

        return np.array(train), np.array(label)


    def getTumorPatchs(self, images, radius, num):
        train, label = [], []
        img_train = images[:4].reshape(4, 240, 240)
        img_label = images[4:].reshape(1, 240, 240)

        grid = 2 * radius
        array = np.ndarray((4, grid, grid))
        cnt = np.argwhere(img_label[0]!=0)
        m = len(cnt)-1
        if(m>0):         
            for i in range(num):
                flag = True
                times = 1
                while (flag and times!=200):
                    idx = random.randint(0, len(cnt)-1)
                    x, y = cnt[idx]
                    l_x, l_y = x - radius, y - radius  # 得到小方格的左上角和右下角坐标
                    r_x, r_y = x + radius, y + radius
                    tmp = img_label[0, l_y:r_y, l_x:r_x]
                    if np.all(tmp!=0):
                        array[:, :, :] = img_train[:, l_y:r_y, l_x:r_x]
                        train.append(array)
                        label.append([2])
                        flag = False
                    times += 1

        return np.array(train), np.array(label)


    def getNormalPatchs(self, images, radius, num):
        #question：num的含义
        train, label = [], []
        img_train = images[:4].reshape(4, 240, 240)
        img_label = images[4:].reshape(1, 240, 240)  
        grid = 2 * radius
        #m = img_label[0]返回图片为0的值argwhere返回满足条件的值
        cnt = np.argwhere(img_label[0]==0)
        array = np.ndarray((4, grid, grid))#创建一个矩阵为（4，26，26）
        for i in range(num):
            flag = True
            times = 1
            while (flag and times!=100):
                idx = random.randint(0, len(cnt)-1)
                x, y = cnt[idx]
                l_x, l_y = x - radius, y - radius  # 得到小方格的左上角和右下角坐标
                r_x, r_y = x + radius, y + radius
                tmpl = img_label[0, l_y:r_y, l_x:r_x]
                tmpt = img_train[1, l_y:r_y, l_x:r_x]
                if (np.all(tmpl==0) and np.all(tmpt!=0) and tmpt.shape[0]==tmpt.shape[1]==grid):
                    array[:, :, :] = img_train[:, l_y:r_y, l_x:r_x]
                    train.append(array)
                    label.append([0])
                    flag = False
                times += 1

        return np.array(train), np.array(label)


def generateData():
    paths = [brain_pielined_dir + s for s in os.listdir(brain_pielined_dir)]
    random.shuffle(paths)#打乱列表元素shuffle
    print(len(paths))#15624张
    new_paths = [paths[random.randint(0, len(paths)-1)] for i in range(5000)]#随机5000张
    print(len(new_paths))
    library = LocDataLibrary(new_paths)
    for t in ['n', 'b', 't']:
        radius = 13
        grid = 2 * radius        
        ldl_ = loc_data_library_(13)
        train_path = ldl_.get_train_save(t)
        label_path = ldl_.get_label_save(t)
        print("train_path:{},label_path:{}".format(train_path,label_path))
        library.loadImages(train_path, label_path, radius=radius, types=t, total=2000)


if __name__ == '__main__':
    ###查看label图片
    # label_path = "D:/BraTS2019_Experiments/DataSets/Npy_Datas/BraTS19_2013_11_1_flair_66.npy"
    # label_io_path = "D:/BraTS2019_Experiments/DataSets/Brain_Pipelined/label_save/BraTS19_2013_11_1_flair_66.png"
    # test = "C:/Users/77419/Desktop/个人图片/QQ截图20200606113420.png"
    # img_path = np.load(label_path)
    # img = img_path[4:,:,:].reshape(240,240)
    # img_io = io.imread(test)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img_io)
    # plt.show()
    ###
    # img_path = 'H:/WorkSpace/SegBrainMRI/Brain_Pipelined/Brats17_2013_0_1_flair_56.png'
    # img = io.imread(img_path).reshape(5, 240, 240).astype('float')[1]
    # result = np.argwhere(img!=0)
    # print(len(result))
    # # array = img[100:105, 50:100]
    # array = img[50:100, 100:105]
    # plt.imshow(img)
    # plt.show()
    generateData()



