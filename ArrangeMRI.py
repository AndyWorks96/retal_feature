# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/5/10
'''
    按照名字的不同将不同的数据整理在一个文件夹下
'''
import os
import cv2
import shutil
import SimpleITK as sitk
import numpy as np 

path_dir = 'D:/BraTS2019_Experiments/19Train/LGG/'
save_dir = 'D:/BraTS2019_Experiments/Raw_dicom/'

# 获取存储目标路径
def getTargetPath(path):
    file_name = path[path.rfind('/')+1:]
    name = file_name[: file_name.rfind('_')]
    save_to_dir = save_dir + name

    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)
    return save_to_dir + '/' + file_name


# 处理主函数
def arrangeFile():
    i = 1
    paths = [path_dir + s for s in os.listdir(path_dir)]
    tmp = []
    for p in paths:
        print('Processing dcm: --> {}'.format(p))
        tmp.append(p)
        
        if i%5 == 0:
            for t in tmp:
                target_path = getTargetPath(t)
                
                shutil.copy(t, target_path)
            print('\n----------------------------')
            tmp = []

        i += 1

if __name__ == '__main__':
    arrangeFile()