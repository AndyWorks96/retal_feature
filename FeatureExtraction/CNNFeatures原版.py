# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/4/8
'''
    通过预训练的神经网络完成特征提取工作，直接加载已经训练好的神经网络模型完成特征提取
    参考自：Mlti-scale Convolutional Neural Networks for Lung Nodule Classification
'''
import keras
import numpy as np
from keras.layers import *
from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from glob import glob
import os
import pandas as pd

model_save = './cnn_features_model.h5'
features_save = './cnn_features.h5'

# CNN特征提取
class CNNFeatures(object):
    def __init__(self, model_dir=features_save):
        self.model_dir = model_dir
        self.model = load_model(self.model_dir)

    def getFeatures(self, array):
        f1 = self.model 
        features = f1.predict(array[None])
        return features


# CNN训练模型
class CNNModelTrains(object):
    def __init__(self, model_save='', features_save='', img_size=[20, 20, 1], n_filters=32, n_features=50):
        #K.set_image_dim_ordering('th')   # 通道优先
        K.set_image_data_format('channels_first')
        print(K.image_data_format())
        self.model_save = model_save
        self.features_save = features_save
        self.img_size =img_size
        self.n_filters = n_filters
        self.n_features = n_features


    def cnnModel(self, img_size, n_filters, n_features):
        img_x, img_y, img_z = img_size[0], img_size[1], img_size[2]

        inputs = Input((img_x, img_y, img_z))

        conv1 = Conv2D(n_filters, (5, 5), strides=1, activation='relu', use_bias=True, padding='same',
                       kernel_initializer='glorot_uniform')(inputs)
        print('conv1 shape = {}'.format(conv1.get_shape()))
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print('pool1 shape = {}'.format(pool1.get_shape()))

        conv2 = Conv2D(2*n_filters, (5, 5), strides=1,activation='relu', use_bias=True, padding='same',
                       kernel_initializer='glorot_uniform')(pool1)
        print('conv2 shape = {}'.format(conv2.get_shape()))
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print('pool2 shape = {}'.format(pool2.get_shape()))

        conv3 = Conv2D(2 * n_filters, (3, 3), strides=1, activation='relu', use_bias=True, padding='same',
                       kernel_initializer='glorot_uniform')(pool2)
        print('conv3 shape = {}'.format(conv3.get_shape()))
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # print('pool3 shape = {}'.format(pool3.get_shape()))

        feature = Dense(n_features, activation='relu', use_bias=True, name='features')(Flatten()(conv3))
        print('feature shape = {}'.format(feature.get_shape()))
        output = Dense(2, activation='sigmoid', name='output')(feature)
        print('output shape = {}'.format(output.get_shape()))

        model = Model(inputs, output)
        feature_model = Model(inputs=model.inputs,
                              outputs=model.get_layer('features').output)

        #sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])
        print(model.summary())
        return model, feature_model


    # def loadData(self, train_path, label_path):
    #     trains = np.load(train_path)
    #     labels = np.load(label_path)
    #     x_train, x_test, y_train, y_test = train_test_split(trains, labels, test_size=0.2)
    #     y_train = to_categorical(y_train)
    #     y_test = to_categorical(y_test)
    #     print(y_train.shape)
    #     return x_train, y_train, x_test, y_test

    def loadData(self, train_path, label_path):

        # Data loading code
        img_paths = glob('/Users/andyworks/python/pythonproject/tensorflow/brats_feature/DataSets/seg_model/dwiImage01/*')
        mask_paths = glob('/Users/andyworks/python/pythonproject/tensorflow/brats_feature/DataSets/seg_model/dwiMask01/*')
        xlsx1_filePath = '/Users/andyworks/python/pythonproject/tensorflow/brats_feature/DataSets/Rectal Outcome.xlsx'
        images = np.zeros([411, 1, 224, 224], dtype=np.float64)
        masks = np.zeros([411, 1], dtype=np.float64)
        data_1 = pd.read_excel(xlsx1_filePath)
        img_name_list = []
        print(len(images))
        for i in range(len(images)):
            img = img_paths[i]
            img_name = img[93:]
            img_npy = img_name[-6]
            if img_npy != '_':
                img_name = img[93:-7]
            else:
                img_name = img[93:-6]
            img_name_list.append(img_name)
            npimage = np.load(img_paths[i])
            npimage = npimage.transpose((2, 0, 1))
            data = data_1.T
            d = data[3]
            dd = data[3][0]
            ddd = data[3][1]

            for ix in range(43):
                if data[ix][0] == img_name:
                        masks[i][0] = data[ix][1]

            for idx in range(images.shape[2]):
                for idy in range(images.shape[3]):
                    images[i, 0, idx, idy] = npimage[0, idx, idy]

        df = pd.DataFrame(img_name_list, columns=['label_name'])

        # 保存到本地excel
        # df.to_excel('./label_name_list.xlsx', index=False)

        # trains = np.load(train_path)
        # labels = np.load(label_path)
        trains = images
        labels = masks
        x_train, x_test, y_train, y_test = train_test_split(trains, labels, test_size=0.2)
        # x_train, x_test = train_test_split(images, test_size=0.2)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        print(y_train.shape)
        return x_train, y_train, x_test, y_test
        # return x_train, x_test

    def fitModel(self, train_path, label_path):
        x_train, y_train, x_test, y_test = self.loadData(train_path, label_path)
         
        img_size = self.img_size
        n_filters = self.n_filters
        n_features = self.n_features
        f, f1 = self.cnnModel(img_size, n_filters, n_features)
        epochs = 10
        batch_size = 32
        training = f.fit(x_train, y_train,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          batch_size=batch_size)
        name = "cnn_features_trains"
        fig = plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.plot(training.history['accuracy'])
        # plt.plot(training.history['val_accuracy'])
        # plt.title('Accuracy of ' + name)
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper right')
        # plt.grid()
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(training.history['loss'])
        # plt.plot(training.history['val_loss'])
        # plt.title('Loss of ' + name)
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper right')
        # plt.grid()
        # plt.savefig('./Runed_Result/Loc_figure/train_26.png')
        # plt.show()
        if self.model_save != '':
            f.save(self.model_save)
        if self.features_save != '':
            f1.save(self.features_save)


    def valModel(self, array):
        f = load_model(self.model_save)
        f1 = load_model(self.features_save)
        pred = f.predict(array)
        print('The predict is: {}'.format(pred))
        print('The ground truth is: {}'.format(pred))
        pred_f1 = f1.predict(array)
        print('The features is : {}'.format(pred_f1))


if __name__ == '__main__':
    # train_path = "/Users/andyworks/python/pythonproject/tensorflow/brats_feature/DataSets/seg_model/patchs/xgb_patch_train_10.npy"
    # label_path = "/Users/andyworks/python/pythonproject/tensorflow/brats_feature/DataSets/seg_model/patchs/xgb_patch_label_10.npy"
    train_path = glob('/Users/andyworks/python/pythonproject/tensorflow/brats_feature/DataSets/seg_model/dwiImage01/*')
    label_path = glob('/Users/andyworks/python/pythonproject/tensorflow/brats_feature/DataSets/seg_model/dwiMask01/*')
    # train_path = "/Users/andyworks/python/pythonproject/tensorflow/brats_feature/DataSets/seg_model/patchs/BraTS19_image.npy"
    # label_path = "/Users/andyworks/python/pythonproject/tensorflow/brats_feature/DataSets/seg_model/patchs/BraTS19_mask.npy"
    # cnn_model_train = CNNModelTrains(model_save, features_save,img_size=[160, 160, 4], n_features=128, n_filters=16)
    cnn_model_train = CNNModelTrains(model_save, features_save, img_size=[1, 224, 224], n_features=128, n_filters=32)
    cnn_model_train.fitModel(train_path, label_path)



    model, feature_model = cnn_model_train.cnnModel(img_size=[1, 224, 224], n_features=128, n_filters=32)
    model.save('../cnn_features_model.h5')
