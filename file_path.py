# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/6/3
'''
    管理所有.py文件的路径信息(包括了每个python文件下的所有路径，均使用一个类来控制)
'''
# radius = 13
workSpace_dir = '../DataSets/'

# brain_pieline
class brain_pieline(object):
    def __init__(self):
        # self.path = workSpace_dir + '/PycharmCode/SegBrainMRI/DataSets/'
        self.path = '../DataSets/Raw_dicom/'
        # self.dcm_dir = 'H:/DataSets/MRIDataSets/LGG_dcm/'
        self.dcm_dir = '../DataSets/Raw_dicom/'
        self.train_save = workSpace_dir + 'Brain_Pipelined/train_save/'
        self.label_save =workSpace_dir + 'Brain_Pipelined/label_save/'



# data_library
class data_library(object):
    def __init__(self):
        pass



# loc_data_library
class loc_data_library_(object):
    def __init__(self, radius=13):
        grid = 2*radius
        root_dir = '../DataSets/loc_model/'
        self.radius = radius
        self.grid = grid
        self.brain_pielined_dir = '../DataSets/Brain_Pipelined/train_save/'
        self.patch_save_dir = root_dir + '/patchs/' + 'loc_patch_'
        
    def get_train_save(self, types):
        self.train_save = self.patch_save_dir + 'train_' + str(self.grid) + '_' + types + '.npy'
        return self.train_save
    def get_label_save(self, types):
        self.label_save = self.patch_save_dir + 'label_' + str(self.grid) + '_' + types + '.npy'
        return self.label_save

# loc_tumor
class loc_tumor_(object):
    def __init__(self, radius=13):
        grid = 2*radius
        root_dir = '../DataSets/loc_model/'
        self.loc_model_path = root_dir + '/ModelWeights/' + 'loc_ord_model_' + str(grid) + '.h5'
        self.loc_patch_scale_path = root_dir + '/ModelWeights/' + 'loc_patch_scale_' + str(grid) + '.npy'




# seg_data_library
class seg_data_library_(object):
    def __init__(self, radius=13):
        grid = 2*radius
        root_dir = '../DataSets/seg_model/'
        self.radius = radius
        self.grid = grid
        #self.dcm_root = '../DataSets/LGG_dcm/'
        self.dcm_root = '../DataSets/Raw_dicom/'
        self.raw_save_dir = '../DataSets/Npy_Datas/'    # dcm转化为npy的存储目录
        self.train_dir = '../DataSets/Brain_Pipelined/train_save/'    # 训练数据的x的读取目录[4, 240, 240]
        self.label_dir = '../DataSets/Npy_Datas/'   # 训练数据的y的读取目录[1, 240, 240]

        self.patch_train_path = root_dir + '/patchs/' + 'xgb_patch_train_' + str(grid) + '.npy'
        self.patch_label_path = root_dir + '/patchs/' + 'xgb_patch_label_' + str(grid) + '.npy'
        self.patch_scale_path = root_dir + '/ModelWeights/' + 'xgb_patch_scale_' + str(grid) + '.npy'    # 缩放patch的Min-Max保存路径
        self.excel_raw_path = root_dir + '/features/' + 'xgb_excel_raw_' + str(grid) + '.xlsx'    # 存储原始未缩放特征的数据路径


# seg_tumor
class seg_tumor_(object):
    def __init__(self, radius=13):
        grid = 2*radius
        root_dir = '../DataSets/seg_model/'    
        self.excel_raw_path = root_dir + '/features/' + 'xgb_excel_raw_' + str(grid) + '.xlsx'   # 原始特征的excel数据路径
        self.excel_scale_path = root_dir + '/ModelWeights/' + 'xgb_excel_scale_' + str(grid) + '.xlsx'    # 缩放excel的Min-Max保存路径
        self.excel_normed_path = root_dir + '/features/' + 'xgb_excel_normed_' + str(grid) + '.xlsx'
        self.seg_model_path = root_dir + '/ModelWeights/' +'xgb_seg_model_' + str(grid) + '.h5'


# CNN featurs
class cnn_features_(object):
    def __init__(self, radius=0):
        root_dir = '../DataSets/seg_model/ModelWeights/Featrues_model/'
        add_name = ''
        if radius != 0:
            add_name = '_' + str(radius)
        self.model_save = root_dir + '/cnn_model' + add_name + '.h5'
        self.features_save = root_dir + '/cnn_features_model' + add_name + '.h5'


# feature_selection
class seg_features_(object):
    def __init__(self, radius=0):
        root_dir = '../DataSets/seg_model/ModelWeights/Featrues_select/'
        add_name = ''
        if radius != 0:
            add_name = '_' + str(radius)
        self.l1_path = root_dir + '/l1_model' + add_name + '.h5'
        self.l1l2_path = root_dir + '/l1l2_model' + add_name + '.h5'
        self.pca_path = root_dir + 'pca_model' + add_name + '.h5'