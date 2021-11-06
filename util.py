# _*_ encoding:utf-8 _*_ 
# Author: Lg
# Date: 19/5/20
'''
'''
import os
import scipy.misc
import numpy as np 
from skimage import io 
from glob import glob

class Utils(object):
    def __inti__(self):
        pass

    # 根据文件名匹配相应的labels
    @staticmethod
    def matchPathWithLabels(label_dir, file_path):
        name = file_path[file_path.rfind('/')+1:file_path.rfind('.')]
        print(name)
        label = glob(label_dir + name + '*')[0]
        return label


    # 统计label中的类别信息
    @staticmethod
    def statisticLabels(label_path, label_idx):
        labels = np.load(label_path)
        num = np.sum(labels==label_idx)
        return num


    # 显示ground_truth和segmentation的结果
    def showGtandSeg(self, segmentation, ground_truth, save_path):
        seg_path = './tenp/util_seg.jpg'
        gt_path = './temp/util_gt.jpg'
        scipy.misc.toimage(seg_path).save(segmentation)
        scipy.misc.toimage(gr_path).save(ground_truth)
        seg = io.imread(seg_path)
        gt = io.imread(gt_path)

        xs1, ys1 = np.where(segmentation==2)   # 1       分割结果
        xs2, ys2 = np.where(segmentation==3)   # 2
        xs3, ys3 = np.where(segmentation==4)   # 3
        for i in range(len(xs1)):
            seg[xs1[i], ys1[i], :] = [255, 0, 0]
        for i in range(len(xs2)):
            seg[xs2[i], ys2[i], :] = [0, 255, 0]
        for i in range(len(xs3)):
            seg[xs3[i], ys3[i], :] = [0, 0, 255]

        xg1, yg1 = np.where(ground_truth==1)   # 1       金标准结果
        xg2, yg2 = np.where(ground_truth==2)   # 2
        xg3, yg3 = np.where(ground_truth==3)   # 3
        for i in range(len(xg1)):
            gt[xg1[i], yg1[i], :] = [255, 0, 0]
        for i in range(len(xg2)):
            gt[xg2[i], yg2[i], :] = [0, 255, 0]
        for i in range(len(xg3)):
            gt[xg3[i], yg3[i], :] = [0, 0, 255]

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(seg)
        plt.title('Segmentation result')

        plt.subplot(1, 2, 2)
        plt.imshow(gt)
        plt.title('ground truth')
        plt.savefig(save_path)
        plt.show()

    # 计算所有数据的缩放值Min-Max
    @staticmethod
    def calcuteALLScale01(brain_piplined_dir):
        paths = [brain_piplined_dir + '/' + s for s in os.listdir(brain_piplined_dir)]
        brain_piplind_img = []
        for p in paths:
            tmp_img = io.imread(p).reshape(5, 240, 240).astype('float')
            brain_piplind_img.append(tmp_img)
        brain_piplind_img = np.array(brain_piplind_img)
        print(brain_piplind_img.shape)

        all_patch_scale_model = []
        for i in range(4):
            min_val = brain_piplind_img[:, i, :, :].min()
            max_val = brain_piplind_img[:, i, :, :].max()
            all_patch_scale_model.append([min_val, max_val])
            if min_val != max_val:
                brain_piplind_img[:, i, :, :] -= min_val
                brain_piplind_img[:, i, :, :] *= (255.0/(max_val - min_val))
        all_patch_scale_model = np.array(all_patch_scale_model)
        print(all_patch_scale_model)
        np.save('./DataSets/seg_model/xgb_all_patch_scale.npy', all_patch_scale_model)

    @staticmethod
    def calcuteALLScale02(brain_piplined_dir):
        paths = [brain_piplined_dir + '/' + s for s in os.listdir(brain_piplined_dir)]
        all_patch_scale_model = [[999, -1],
                                 [999, -1],
                                 [999, -1],
                                 [999, -1]]
        time = 1#定义次数
        for p in paths:
            print("times:{}/{}".format(time,len(paths)))
            time += 1
            tmp_img = io.imread(p).reshape(5, 240, 240).astype('float')
            for i in range(4):
                min_val = tmp_img[i].min()#找出前五层每一层的最大最小值
                max_val = tmp_img[i].max()
                if min_val < all_patch_scale_model[i][0]:
                    all_patch_scale_model[i][0] = min_val
                if max_val > all_patch_scale_model[i][1]:
                    all_patch_scale_model[i][1] = max_val
             #找出全部图像像素的最小最大值
        print(all_patch_scale_model)
        np.save('../DataSets/seg_model/xgb_all_patch_scale.npy', all_patch_scale_model)
        print("already save to ../DataSets/seg_model/xgb_all_patch_scale.npy")



if __name__ == '__main__':
    brain_piplind_dir = '../DataSets/Brain_Pipelined/train_save'
    Utils.calcuteALLScale02(brain_piplind_dir)
