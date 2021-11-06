# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/5/17
'''
'''
import random
import numpy as np 
from glob import glob
# from skimage import io 
from scipy import misc
# from skimage.filters.rank import entropy
# from skimage.morphology import disk


class DataLibrary(object):
    def __init__(self):
        pass

    # 保存打乱的图像路径
    def saveImgsPaths(self, imgs_paths):
        name = './imgs_paths.txt'
        random.shuffle(imgs_paths)
        with open(name, 'a') as file:
            for p in imgs_paths:
                file.write(p + '\n')
        return name

    # 读取图像路径
    def readImgsPaths(self, name):
        imgs_paths = []
        with open(name, 'r') as file:
            content = file.readlines()
            for c in content:
                c = c.strip('\n')
                c = c.strip('\r')
                imgs_paths.append(c)
        return imgs_paths


    # 分批获取图像
    def getTrainBatch(self, t_path, l_path, batch_size, start=0, end=5000):
        while True:
            for i in range(start, end, batch_size):
                x, y = self.getImgs(t_path[i: i+batch_size], l_path[i: i+batch_size])
                yield({'input_1': x}, {'softmax_1':y})
                

    # 读取图像
    def getImgs(self, t_path, l_path):
        trains = []   # 训练数据
        labels = []   # 标签
        for i in range(len(t_path)):
            # img = io.imread(paths[i]).reshape(5, 240, 240).astype('float')
            imgt = misc.imread(t_path[i]).reshape(4, 240, 240).astype('float')
            trains.append(imgt)
            imgl = misc.imread(l_path[i]).reshape(1, 240, 240).astype('float')
            labels.append(imgl)
        return np.array(trains), np.array(labels)


    # 生成训练数据
    def getTrainingPatchs(self, paths, num_patchs, patch_size, classes=[0, 1, 2, 3, 4]):
        h = patch_size[0]
        w = patch_size[1]
        patchs, labels = [], []
        per_class = num_patchs / len(classes)    # 获取每个类的总数
        for c in classes:
            p, l = self.getPatchs(paths, per_class, patch_size, c)
            patchs.append(p)
            labels.append(l)
        return np.array(patchs).reshape(num_patchs, 4, h, w), np.array(labels).reshape(num_patchs)



    # 根据类别标注class_num和样本总数num_patchs获取patch
    def getPatchs(self, paths, num_patchs, patch_size, class_num):
        h = patch_size[0]
        w = patch_size[1]
        patchs, labels = [], np.full(num_patchs, class_num, 'float')   # 根据类别信息和总数直接生成标签
        ct = 0
        while ct < num_patchs:
            img_path = random.choice(paths)
            imgs = io.imread(img_paths).reshape(5, 240, 240).astype('float')
            label = imgs[-1:]
            if len(np.argwhere(label == class_num)) < 10:    # 类别太少时直接跳过
                continue
            img = imgs[:4]
            p = random.choice(np.argwhere(label == class_num))    # 随机选取一个属于该类别的像素点[x, y]
            p_ix = ( p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2) )    # 生成patch的范围[x1, x2, y1, y2]
            patch = np.array(i[p_ix[0]: p_ix[1], p_ix[2]: p_ix[3]] for i in img)    # 选取patch

            if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (h*w):
                continue

            patchs.append(patch)
            ct += 1
        return np.array(patchs), labels


    # 基于熵最大原则选取patch，主要针对边缘区域
    def patchsByEntropy(self, paths, num_patchs, patch_size):
        h = patch_size[0]
        w = patch_size[1]
        patchs, labels = [], []
        ct = 0
        while ct < num_patchs:
            img_path = random.choice(paths)
            imgs = io.imread(img_paths).reshape(5, 240, 240).astype('float')
            label = imgs[-1:]
            if len(np.unique(label)) == 1:    # 如果只包含一种类别则舍去
                continue
            patch = imgs[:4]
            l_ent = entropy(label, disk(h))    # 计算label在h为半径的圆盘范围内的熵，输出为一个相同大小的图像
            top_ent = np.percentile(l_ent, 90)    # 计算90%的分位数

            if top_ent == 0:
                continue

            higest = np.argwhere(l_ent >= top_ent)    # 随机在前10%的最大熵中挑一个
            p_s = random.sample(highest, 3)    # 从前10%的序列中选取3个随机且独立的元素
            for p in p_s:
                p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
                patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
                if np.shape(patch) != (4, h, w):
                    continue
                patches.append(patch)
                labels.append(label[p[0],p[1]])
            ct += 1

        return np.array(patchs[:num_patchs]), np.array(labels[:num_patchs])
