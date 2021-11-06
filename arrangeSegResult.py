# _*_ encoding:utf-8 _*_

import os
import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob
from evalution_index import EvalutionIndex, SaveEvaltuion
from zloc_seg_tumor2 import calcuteEvaluteIndex


def main():
    se = SaveEvaltuion()
    total_dices = []

    dest_dir = 'E:/BraTS2019_Experiments/DataSets/Brain_numpy/'

    root_dir = 'E:/BraTS2019_Experiments/Runed_Result/new_result/array_fig/'
    file_paths = [root_dir + s for s in os.listdir(root_dir)]

    for path in file_paths:
        name = path[path.rfind('/')+1: path.rfind('_')]
        label_path = glob(dest_dir + name + '*')[0]
        label = np.load(label_path)

        segmentation = np.load(path)[:, 240:]

        seg_array = np.concatenate((label, segmentation), axis=1)
        np.save('./Runed_Result/new_result1/array_fig/' + name + '.npy', seg_array)

        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        axs[0].imshow(label)
        axs[0].set_title('Ground truth')
        axs[1].imshow(segmentation)
        axs[1].set_title('Seg result')
        fig.suptitle(name, fontsize=14)
        plt.savefig('./Runed_Result/new_result1/jpg_fig/' + name + '.jpg')
        # plt.savefig('./temp/' + name + '.jpg')
        evaluate_score = calcuteEvaluteIndex(label, segmentation)
        print('The img >{}<:'.format(name))
        print('\t 1 dice: {}\n\t 2 sensitivity: {}\n\t 3 spensifity: {}\n'.format(evaluate_score[0], evaluate_score[1], evaluate_score[2]))
    
        se.addSeries(name, evaluate_score)
        # if (k % 3 == 0):
        #     se.save('./Runed_Result/Seg_figure/seg_result.csv')
        se.save('./Runed_Result/new_result/seg_result8.csv')


if __name__ == '__main__':
    main()