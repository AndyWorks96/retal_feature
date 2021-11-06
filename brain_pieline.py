# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/5/15
'''
    预处理，对.dcm图像进行归一化，并保存为.png
'''
import os
import time
import random 
import warnings
import scipy.misc
import SimpleITK as sitk #读取文件
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image#banduo-add 用于替换scipy
from math import ceil
from skimage import exposure #直方图
from glob import glob #返回目录下文件
from skimage import io 
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)


num = 0
from file_path import brain_pieline
bp = brain_pieline()
path = bp.path
train_save = bp.train_save
label_save = bp.label_save

class BrainPipeline(object):
    def __init__(self):
        pass

    # 读取切片
    def readScans(self, path):
        print('Loading scans...')

        slices_by_mode = np.zeros((5, 155, 240, 240))
        slices_by_slices = np.zeros((155, 5, 240, 240))

        flair = glob(path + '/*flair.dcm')
        t1 = glob(path + '/*t1.dcm')
        t1_ce = glob(path + '/*t1ce.dcm')
        t2 = glob(path + '/*t2.dcm')
        seg = glob(path + '/*_seg.dcm')
        scans = [flair[0], t1[0], t1_ce[0], t2[0], seg[0]]    
        print("scans:{}".format(scans[0]))

        self.patient_num = scans[0][scans[0].rfind('\\')+1: scans[0].rfind('.')]
        self.patient_num = self.patient_num[43:]
        print("patient_num:{}".format(self.patient_num))       
        for scan_idx in range(5):
            slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
        for mode_idx in range(slices_by_mode.shape[0]):
            for slice_idx in range(slices_by_mode.shape[1]):
                slices_by_slices[slice_idx][mode_idx] = slices_by_mode[mode_idx][slice_idx]   # reshape数据为CNN的输入

        return slices_by_mode, slices_by_slices


    # 处理一个病人的数据
    def normSlices(self, slices_by_slices):
        print('Normalizing slices...')
        normed_slices = np.zeros((155, 5, 240, 240))
        for slice_idx in range(155):
            normed_slices[slice_idx][-1] = self.convertLabel(slices_by_slices[slice_idx][-1])   # 将数据标签放在最后一维
            # normed_slices[slice_idx][-1] = slices_by_slices[slice_idx][-1]    # 不对数据标签进行转化
            for mode_idx in range(4):   # 对除过标签的数据进行归一化
                normed_slices[slice_idx][mode_idx] = self.normalize(slices_by_slices[slice_idx][mode_idx])
        print('Done.')
        return normed_slices


    # 将标签数据进行转化为0和1
    def convertLabel(self, slices):
        mask = slices
        mask[mask!=0] = 1   # 将[1，2，3，4]的标签全部置为0
        # io.imsave('./{}_{}.png'.format(int(time.time()), int(random.random()*1000)), mask)
        return mask


    # 数据标准化
    def normalize(self, slices):
        b, t = np.percentile(slices, (0.5, 99.5))
        slices = np.clip(slices, b, t)   # 将slice的数据限制在b和t之间
        if np.std(slices) == 0:
            slices = slices 
        else:
            slices = (slices - np.mean(slices)) / np.std(slices)
            slices[slices<0] = 0
        return slices
        #return self.doHistEqualiztion(slices)
    


    # 保存归一化后的切片
    def saveNormSlices(self, normed_slices, train_path='', label_path=''):
        patient_num = self.patient_num
        global num
        print('Saving scans for patient {}...'.format(patient_num))
        for slice_idx in range(155):   
            strip = normed_slices[slice_idx].reshape(1200, 240)   # 4*240=960
            tmp = normed_slices[slice_idx, -1, :, :]
            result = np.argwhere(tmp!=0)
            if len(result)>600:
                num += 1
                print(patient_num)
                ###########
                # io.imsave(train_path + '/{}_{}.png'.format(patient_num, slice_idx), strip) 
                # print(np.min(normed_slices[slice_idx][-1]), np.max(normed_slices[slice_idx][-1]))
                # Image.fromarray(255*normed_slices[slice_idx][-1]).convert('L').save(label_path + '/{}_{}.png'.format(patient_num, slice_idx))
                ###李广1###
                #scipy.misc.toimage(normed_slices[slice_idx][-1], cmin=0, cmax=255).save(label_path + '/{}_{}.png'.format(patient_num, slice_idx))
                #strip = strip.reshape(5, 240, 240)
                ###生成np文件###
                train_path_np = "../DataSets/Brain_Pipelined/numpy"
                np.save(train_path_np + '/{}_{}.npy'.format(patient_num, slice_idx), tmp)
                
    # 做直方图均衡化
    def doHistEqualiztion(self, array, height=0, width=0):
        if height == 0 & width == 0:
            equalized = exposure.equalize_hist(array)
        else:
            w,h = array.shape
            equalized = np.ndarray((w, h))
            for i in range(ceil(w/width)):
                for j in range(ceil(h/height)):
                    equalized[i*width:i*width + width, j*height:h*height + height] = exposure.equalize_hist(array[i*width:i*width + width, j*height:h*height + height])

        return equalized

    # 保存归一化后的切片
    def saveOrigSlices(self, orig_slices, path):
        patient_num = self.patient_num
        print('Saving scans for patient {}...'.format(patient_num))
        for slice_idx in range(155):
            strip = orig_slices[slice_idx].reshape(1200, 240)   # 5*240=1200
            if np.max(strip) != 0:
                strip /= np.max(strip)
            io.imsave('../DataSets/png' + '/{}_{}.png'.format(patient_num, slice_idx), strip)
        

    # 保存为.npy
    def saveSlicesToNpy(self, orig_slices, path):
        patient_num = self.patient_num
        print('Saving scans for patient {}...'.format(patient_num))
        for slice_idx in range(155):
            strip = orig_slices[slices_idx]   # 5*240=1200
            if np.max(strip) != 0:
                strip /= np.max(strip)
            if np.min(strip) <= -1:
                strip /= abs(np.min(strip))
            np.save(path + '/{}_{}.npy'.format(patient_num, slices_idx), strip)
            



# 保存处理完的数据
def savePatient(types, path, train_save, label_save):
    brainp = BrainPipeline()
    modes, slices = brainp.readScans(path)
    print("model:{},slices:{}".format(modes.shape,slices.shape))
    if types == 'orig':
        print('*****'+path)
        brainp.saveOrigSlices(slices, path)
    if types == 'npy':
        brainp.saveSlicesToNpy(slices, path)
    if types == 'norm':
        #slices = brainp.normSlices(slices)#归一化切片
        brainp.saveNormSlices(slices, train_save, label_save)
    if types == 'null':
        print('There is nothing to do ...')
 

def main():
    dcm_dir = '../DataSets/Raw_dicom/'
    dirs = [dcm_dir + s for s in os.listdir(dcm_dir)]
    for p in dirs:
        # train_save = 'D:/BraTS2019_Experiments/DataSets/Brain_Pipeline/'
        # #train_save = 'D:/BraTS2019_Experiments/DataSets/Brain_numpy/'
        # label_save = 'D:/WorkSpace/SegBrainMRI/Brain_Pipelined/'
        # types = 'norm'
        if p != '../DataSets/Raw_dicom/.DS_Store':
            types = 'orig'
            savePatient(types, p, train_save, label_save)
        # break


if __name__ == '__main__': 
    # xx = np.load("D:/BraTS2019_Experiments/DataSets/Brain_Pipelined/train_save_np/BraTS19_2013_0_1_flair_46.npy")
    # io.imshow(xx)
    # plt.show()
    # yy = io.imread("D:/BraTS2019_Experiments/DataSets/Brain_Pipelined/train_save/BraTS19_2013_0_1_flair_47.png")
    # io.imshow(yy)
    # plt.show()
    # zz = "D:/BraTS2019_Experiments/DataSets/Brain_Pipelined/train_save/test.png"
    # zz = io.imread(zz)
    # io.imshow(zz)
    # plt.show()

    main()
    
    # path = "D:/BraTS2019_Experiments/DataSets/Brain_Pipelined/train_save/BraTS19_2013_0_1_flair_55.png"
    # img = io.imread(path)
    # plt.imshow(img)
    # plt.show()
    # ##test##
    # dirs = './temp/'
    # paths = glob(dirs + '*')
    # for p in paths:
    #     img = io.imread(p).astype('float')
    #     plt.imshow(img)
    #     plt.show()
        # break
