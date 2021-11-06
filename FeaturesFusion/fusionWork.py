# _*_ encoding:utf-8 _*_
# 采用三种方法判断融合效果：1.相关性衡量；2.平均值曲线；3.进行聚类分析

import warnings
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
import seaborn as sns
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_mutual_info_score, completeness_score, homogeneity_score, v_measure_score
warnings.filterwarnings("ignore")


class FeatureFusion(object):
    def __init__(self):
        self.raw_file_path = ''
        self.normed_file_path = '/Users/manfestain/Downloads/FeaureFusion/xgb_excel_normed_8.xlsx'
        self.fusion_file_path = './DEselect_result/DE_features_bk.xlsx'
        self.readFile()


    def readFile(self):
        data = pd.read_excel(self.normed_file_path)
        columns = data.columns
        columns = [name for name in columns if 'zLabel' not in name]
        cnn_columns = [name for name in columns if 'cnnf_' in name]
        rad_columns = [name for name in columns if 'cnnf_' not in name]
        mixed_columns = columns

        self.label = data['zLabel']
        self.cnn_features = data[cnn_columns]
        self.radiomics_features = data[rad_columns]
        self.selected_features = pd.read_excel(self.fusion_file_path)
        self.mixed_features = data[mixed_columns]


    def clusterMeasure(self):

        def clusterData(data):
            y = KMeans(n_clusters=4).fit(data)
            table = pd.concat([data, pd.Series(y.labels_, index=data.index)], axis=1)
            table.columns = list(data.columns) + ['zlabel']
            return table

        def getPlotData(data):
            tsne = TSNE()
            tsne.fit_transform(data)
            tsne = pd.DataFrame(tsne.embedding_, index=data.index)
            colors = ['r', 'b', 'g', 'k']
            markers = ['.', '^', 'x', '+']
            for i in range(4):
                d = tsne[data['zlabel']==i]
                plt.plot(d[0], d[1], colors[i]+markers[i], markersize=3, markeredgewidth=.5, label=i+1)
            plt.legend(loc=1)
            plt.xticks([])
            plt.yticks([])

        def getClusterResult(data, labels):
            s1 = silhouette_score(data, labels)
            # s2 = calinski_harabasz_score(data, labels)
            s3 = davies_bouldin_score(data, labels)
            s4 = adjusted_mutual_info_score(self.label, labels)
            s5 = completeness_score(self.label, labels)
            s6 = homogeneity_score(self.label, labels)
            s7 = v_measure_score(self.label, labels)
            return [s1, s3, s4, s5, s6, s7]

        fig = plt.figure()
        cnn_table = clusterData(self.cnn_features)
        cnn_score = getClusterResult(self.cnn_features, cnn_table['zlabel'])
        getPlotData(cnn_table)
        plt.title('CNN Features')
        # plt.savefig('./temp/cluster_cnn.png', dpi=300)
    
        fig = plt.figure()
        radiomics_table = clusterData(self.radiomics_features)
        radiomics_score = getClusterResult(self.radiomics_features, radiomics_table['zlabel'])
        getPlotData(radiomics_table)
        plt.title('Radiomics Features')
        plt.savefig('./temp/cluster_radiomics.png', dpi=300)
        
        fig = plt.figure()
        fusion_table = clusterData(self.selected_features)
        fusion_score = getClusterResult(self.selected_features, fusion_table['zlabel'])
        getPlotData(fusion_table)
        plt.title('Selected Features')
        plt.savefig('./temp/cluster_selected.png', dpi=300)

        fig = plt.figure()
        mixed_table = clusterData(self.mixed_features)
        mixed_score = getClusterResult(
            self.mixed_features, mixed_table['zlabel'])
        getPlotData(mixed_table)
        plt.title('Mixed Features')
        plt.savefig('./temp/cluster_mixed.png', dpi=300)

        # plt.show()

        names = ['silhouette_score', 'davies_bouldin_score', 'adjusted_mutual_info_score', 'completeness_score', 'homogeneity_score', 'v_measure_score']
        dataFrame = pd.DataFrame(columns=names)
        dataFrame.loc[0] = cnn_score
        dataFrame.loc[1] = radiomics_score
        dataFrame.loc[2] = fusion_score
        dataFrame.loc[3] = mixed_score
        dataFrame.insert(0, 'func', ['cnn', 'radiomics', 'selected', 'mixed'])
        dataFrame.to_csv('./temp/cluster_evaluate_result.csv', index=False)



def main():
    # tips = pd.read_csv('/Users/manfestain/Downloads/seaborn-data-master/tips.csv')
    # relatedMeasure(tips, tips)
    features_fusion = FeatureFusion()
    # print(features_fusion.label)
    # features_fusion.relatedMeasure()
    features_fusion.clusterMeasure()
    # features_fusion.meanLineMeasure()



if __name__ == '__main__':
    main()
    
