# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/5/15
'''
    预分割肿瘤（定位）
    1，先训练一个肿瘤定位模型
    2，根据训练的模型进行肿瘤定位，返回肿瘤的边界区域
'''
import os
import cv2
import time
import random 
import datetime
import warnings
import numpy as np 
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from skimage import io 
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
from keras.layers import merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Softmax, concatenate, Dense, Flatten
from keras.models import *
from keras.optimizers import *
from keras.regularizers import L1L2
from keras.utils import to_categorical, plot_model
from file_path import loc_tumor_
warnings.filterwarnings('ignore')

class LocationTumor(object):
    def __init__(self, save_path='', epochs=10, batch_size=16):
        #K.set_image_dim_ordering('th')   # 通道优先
        K.set_image_data_format('channels_first')
        print(K.image_data_format())
        self.save_path = save_path
        self.epochs = epochs
        self.batch_size = batch_size



    # Github中的双路径模型（全局和局部）
    def compileGlobalModel(self, n_dims, n_filters=64):
        print('Comiling two-path model ...')
        img_z, img_x, img_y = n_dims[0], n_dims[1], n_dims[2]

        inputs = Input((img_z, img_x, img_y))

        conv1 = Conv2D(64, (7, 7), strides=1, activation='relu', use_bias=True, W_regularizer=L1L2(l1=0.01, l2=0.01), 
                       kernel_initializer='glorot_uniform')(inputs)
        print('conv1 shape = {}'.format(conv1.get_shape()))
        pool1 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1))(conv1)
        print('pool1 shape = {}'.format(pool1.get_shape()))
        drop1 = Dropout(0.5)(pool1)

        conv2 = Conv2D(64, (3, 3), strides=1, activation='relu', use_bias=True, W_regularizer=L1L2(l1=0.01, l2=0.01),
                       kernel_initializer='glorot_uniform')(drop1)
        print('conv2 shape = {}'.format(conv2.get_shape()))
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(conv2)
        print('pool2 shape = {}'.format(pool2.get_shape()))

        conv11 = Conv2D(160, (13, 13), strides=1, activation='relu', use_bias=True, W_regularizer=L1L2(l1=0.01, l2=0.01), 
                       kernel_initializer='glorot_uniform')(inputs)
        print('conv11 shape = {}'.format(conv11.get_shape()))
        drop11 = Dropout(0.5)(pool2)
        drop12 = Dropout(0.5)(conv11)
        cont11 = concatenate([drop11, drop12], axis=1)
        conv3 = Conv2D(5, (21, 21), strides=1, activation='relu', use_bias=True, W_regularizer=L1L2(l1=0.01, l2=0.01), padding='same',
                       kernel_initializer='glorot_uniform')(cont11)
        print('conv22 shape = {}'.format(conv3.get_shape()))
        flatten = Flatten()(conv3)
        outputs = Dense(2, activation='sigmoid')(flatten)
        model = Model(inputs, outputs)
        sgd = SGD(lr=0.005, decay=0.1, momentum=0.9)
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        print(model.summary())
        print('Compile model done!')
        return model



    # 使用金字塔池化的模型
    def compileStackModel(self, n_dims, n_filters=64):
        img_z, img_x, img_y = n_dims[0], n_dims[1], n_dims[2]

        inputs = Input((img_z, img_x, img_y))
        conv1 = Conv2D(n_filters, (3, 3), strides=1, activation='relu', use_bias=True, W_regularizer=L1L2(l1=0.01, l2=0.01), padding='same',
                       kernel_initializer='glorot_uniform')(inputs)
        print('conv1 shape = {}'.format(conv1.get_shape()))
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print('pool1 shape = {}'.format(pool1.get_shape()))
        drop1 = Dropout(0.2)(pool1)

        conv2 = Conv2D(2*n_filters, (3, 3), strides=1, activation='relu', use_bias=True, W_regularizer=L1L2(l1=0.01, l2=0.01), padding='same',
                       kernel_initializer='glorot_uniform')(drop1)
        print('conv2 shape = {}'.format(conv2.get_shape()))
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print('pool2 shape = {}'.format(pool2.get_shape()))

        conv12 = Conv2D(n_filters, (7, 7), strides=1, activation='relu', use_bias=True, W_regularizer=L1L2(l1=0.01, l2=0.01), padding='same',
                       kernel_initializer='glorot_uniform')(inputs)
        print('conv12 shape = {}'.format(conv12.get_shape()))
        drop12 = Dropout(0.3)(conv12)
        print('drop12 shape = {}'.format(drop12.get_shape()))
        #drop121 = Dropout(0.3)(drop12)
        #print('drop121 shape = {}'.format(drop121.get_shape()))
        cont12 = concatenate([pool2, drop12], axis=1)
        conv22 = Conv2D(2*n_filters, (3, 3), strides=1, activation='relu', use_bias=True, W_regularizer=L1L2(l1=0.01, l2=0.01), padding='same',
                       kernel_initializer='glorot_uniform')(cont12)
        print('conv22 shape = {}'.format(conv22.get_shape()))
        
        outputs = Dense(2, activation='sigmoid')(Flatten()(conv22))
        model = Model(inputs, outputs)
        sgd = SGD(lr=0.005, decay=0.1, momentum=0.9)
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        print(model.summary())
        print('Compile model done!')
        return model



    # 普通的卷积网络模型（CT中使用的普通模型）
    def compileOrdinaryModel(self, n_dims, n_filters=64, n_features=128):
        img_z, img_x, img_y = n_dims[0], n_dims[1], n_dims[2]

        inputs = Input((img_z, img_x, img_y))

        conv1 = Conv2D(n_filters, (7, 7), strides=1, activation='relu', use_bias=True, padding='same',
                       kernel_initializer='glorot_uniform')(inputs)
        print('conv1 shape = {}'.format(conv1.get_shape()))
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print('pool1 shape = {}'.format(pool1.get_shape()))

        conv2 = Conv2D(2*n_filters, (5, 5), strides=1,activation='relu', use_bias=True, padding='same',
                       kernel_initializer='glorot_uniform')(pool1)
        print('conv2 shape = {}'.format(conv2.get_shape()))
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print('pool2 shape = {}'.format(pool2.get_shape()))

        conv3 = Conv2D(4*n_filters, (3, 3), strides=1, activation='relu', use_bias=True, padding='same',
                       kernel_initializer='glorot_uniform')(pool2)
        print('conv3 shape = {}'.format(conv3.get_shape()))
        pool3 = MaxPooling2D(pool_size=(2, 2))(Dropout(0.2)(conv3))
        print('pool3 shape = {}'.format(pool2.get_shape()))

        conv4 = Conv2D(4*n_filters, (3, 3), strides=1, activation='relu', use_bias=True, padding='same',
                       kernel_initializer='glorot_uniform')(pool3)
        print('conv4 shape = {}'.format(conv4.get_shape()))
        
        feature = Dense(n_features, activation='relu', use_bias=True, name='features')(Dropout(0.35)(Flatten()(conv4)))
        print('feature shape = {}'.format(feature.get_shape()))

        output = Dense(3, activation='softmax', name='output')(feature)     # 输出
        print('output shape = {}'.format(output.get_shape()))

        model = Model(inputs, output)
        sgd = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=False)
        model.compile(loss='categorical_crossentropy',   # categorical_crossentropy
                      optimizer='Adam',
                      metrics=['accuracy'])
        print(model.summary())
        return model


    def fitModel(self, model, x_train, y_train):
        # es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        # checkpointer = ModelCheckpoint(filepath="./check/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
        training = model.fit(x_train, y_train,
                             batch_size=self.batch_size,
                             epochs=self.epochs,
                             validation_split=0.1,
                            )
        #self.epochs
        name = "Loc_model_26"
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(training.history['accuracy'])
        plt.plot(training.history['val_accuracy'])
        plt.title('Accuracy of ' + name)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(training.history['loss'])
        plt.plot(training.history['val_loss'])
        plt.title('Loss of ' + name)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.grid()
        plt.savefig('./Runed_Result/Loc_figure/train_26.png')
        plt.show()
        model.save(self.save_path)



    # 根据训练的到的模型进行肿瘤定位
    def locationTumor(self, loc_model_path, scale_model_path='../DataSets/ModelWeights/norm_tumor_Model/scale_model.npy', img_path='', grid=36, radius=18, step=18, loc_save_name=''):
        loc_model = load_model(loc_model_path)
        scale_model = np.load(scale_model_path)

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
                    print(result)
                    mask[l_y:r_y, l_x:r_x] = result + 1
                else:
                    mask[y, x] = 0

        mask_path = '../temp/temp_loc_mask.jpg'
        # plt.imshow(mask)
        # plt.show()
        #scipy.misc.toimage(mask).save(mask_path)
        io.imsave(mask_path,mask)   
        #Image.fromarray(mask).save(mask_path)
        print('The location Model time: {}s'.format(datetime.datetime.now() - starttime))

        img_mask = cv2.imread(mask_path)        # ---------------------------寻找最大轮廓边界
        imgray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        # plt.imshow(imgray)
        # plt.show()
        ret, thresh = cv2.threshold(imgray, 200, 255, 1)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_sorted = sorted(contours, key=lambda x : len(x))
        print(len(contours_sorted))
        cnt = contours_sorted[-1]
        img_mask = cv2.drawContours(label, [cnt], 0, (1000, 0, 0), 2)
        show_img = img[1, :, :]
        show_img[mask==2] = 1
        plt.imshow(label)
        plt.imshow(show_img)
        plt.show()
        

        # # 判断分割目标和实际标签的重合度
        # label[mask==2] = 10000 
        # plt.imshow(img_mask)
        # plt.show()

        return np.array(cnt).squeeze()

        

# 数据归一化
def doNormalize(trains, save_path='./loc_patchs_scale_grid.npy'):
    scale_model = []
    trains = trains.astype('float')
    for i in range(4):
        min_val = trains[:, i, :, :].min()
        max_val = trains[:, i, :, :].max()
        scale_model.append([min_val, max_val])
        if min_val != max_val:
            trains[:, i, :, :] -= min_val
            trains[:, i, :, :] *= (1.0/(max_val - min_val))
    scale_model = np.array(scale_model)
    print(scale_model)
    np.save(save_path, scale_model)
    return trains


# 训练
def train():
    lt_ = loc_tumor_()
    loc_model_path  = lt_.loc_model_path
    loc_patch_scale_path = lt_.loc_patch_scale_path
    # train_save = 'H:/WorkSpace/SegBrainMRI/DataSets/result/'
    # label_save = 'H:/WorkSpace/SegBrainMRI/DataSets/labels/'
    train_save = '../DataSets/loc_model/patchs/'
    label_save = '../DataSets/loc_model/patchs/'

    trains, labels = [], []
    for t in ['n', 't']:
        train_path = train_save + 'loc_patch_train_26_' + t + '.npy'
        label_path = label_save + 'loc_patch_label_26_' + t + '.npy'
        trains.extend(np.load(train_path))
        labels.extend(np.load(label_path))
    trains = np.array(trains)
    labels = np.array(labels)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels = enc.transform(labels).toarray()

    trains = doNormalize(trains, loc_patch_scale_path)   # 归一化
    shuffle = list(zip(trains, labels))   # 打乱数据
    np.random.shuffle(shuffle)

    x_train = np.array([shuffle[i][0] for i in range(len(shuffle))])
    y_train = np.array([shuffle[i][1] for i in range(len(shuffle))])
    print(y_train[:10])
    
    epochs = 20#论文中使用20代 30
    batch_size = 16
    locTumor = LocationTumor(save_path=loc_model_path, epochs=epochs, batch_size=batch_size)
    model = locTumor.compileGlobalModel([4, 26, 26])
    # model = locTumor.compileOrdinaryModel([4, 26, 26])
    locTumor.fitModel(model, x_train, y_train)

# 验证
def validate(img_path):
    radius = 13
    grid = 2*radius
    # loc_model_path = './ModelWeights/loc_model_Global_nt_' + str(grid) + '.h5'
    scale_model_path = '../DataSets/loc_model/ModelWeights/loc_patch_scale_26.npy'
    #scale_model_path = '../DataSets/loc_model/ModelWeights/loc_patch_scale_hgg_26.npy'
    # img_path = './DataSets/Brain_Pipelined/Brats17_TCIA_202_1_flair_91.png'
    loc_model_path = '../DataSets/loc_model/ModelWeights/loc_ord_model_26.h5'
    locTumor = LocationTumor()
    locTumor.locationTumor(loc_model_path, scale_model_path, img_path=img_path, grid=grid, radius=radius, step=radius)


if __name__ == '__main__':
    # loc = LocationTumor()
    # #model = loc.compileStackModel([4, 33, 33])
    # model = loc.compileGlobalModel([4, 26, 26])
    # plot_model(model, to_file='../temp/model.png', show_shapes=True)
    # train()
    # dcm_dir = './DataSets/Brain_Pipelined/'
    #dcm_dir = '../DataSets/Raw_dicom/BraTS19_2013_11_1/'
    dcm_dir = '../DataSets/Brain_Pipelined/train_save/'
    img_paths = [dcm_dir + '/' + s for s in os.listdir(dcm_dir)]
    for x in range(100):
        i = random.randint(0, 4000)
        validate(img_paths[i])
    