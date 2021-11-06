# _*_ encoding:utf-8 _*_
import numpy as np
import scipy.misc
import keras 
import tensorflow as tf
from keras import backend as K

# y_true = np.zeros((2, 1, 240, 240))
# y_pred = np.zeros((2, 1, 240, 240))
# y_true[:, 0, :5, :5] = 1
# weights = np.ndarray(y_true.shape)
# weights[:, :, :] = y_true[:, :, :]
# weights[weights!=0] = 100
# print(y_true)

# loss = K.mean(K.square((y_pred - y_true)**2*(y_true*100)), axis=-1) 
# sess = tf.Session()
# print(sess.run(loss))
# print(loss)
# a = np.ones((240, 240))
# print(([('', a.dtype)]*a.shape[1]).shape)
# print(a.view([('', a.dtype)]*a.shape[1]).shape)
# path ='./tumorMRI/result/Brats17_TCIA_623_1_flair_40.png'
# imgt = scipy.misc.imread(path).reshape(4, 240, 240).astype('float')
# print(imgt.shape)
# import sys
# def main(argv=[20, 20]):
#     parameters = argv[1:]
#     print(parameters)
#     print(type(int(parameters[0])))

# if __name__ == '__main__':
#     main(sys.argv)
# class Foo(object):
#     def __init__(self):
#         self.name = 'this'

#     @staticmethod
#     def printName(self):
#         print('what it is ?')


# Foo.printName()
import matplotlib.pyplot as plt
import numpy as np

mask = np.load('E:/BraTS2019_Experiments/Runed_Result/new_result/array_fig/BraTS19_2013_3_1_flair_72_8.npy')
plt.imshow(mask[:, 240:])
plt.show()
print(mask.shape)


# def f(t):
#     s1 = np.cos(2*np.pi*t)
#     e1 = np.exp(-t)
#     return s1 * e1

# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)
# t3 = np.arange(0.0, 2.0, 0.01)
# img1 = np.ones((240, 240))
# img2 = np.zeros((240, 240))

# fig, axs = plt.subplots(1, 2, constrained_layout=True)
# # axs[0].plot(t1, f(t1), 'o', t2, f(t2), '-')
# # axs[0].set_title('subplot 1')
# # axs[0].set_xlabel('distance (m)')
# # axs[0].set_ylabel('Damped oscillation')
# axs[0].imshow(img1)
# axs[0].set_title('seg tumor')
# fig.suptitle('This is a somewhat long figure title', fontsize=16)

# # axs[1].plot(t3, np.cos(2*np.pi*t3), '--')
# # axs[1].set_xlabel('time (s)')
# # axs[1].set_title('subplot 2')
# # axs[1].set_ylabel('Undamped')
# axs[1].imshow(img2)
# axs[1].set_title('groud truth')
# plt.savefig('./temp/asd.jpg')
# plt.show()

# from skimage import io 

# img_paths = 'E:/BraTS2019_Experiments/DataSets/Brain_Pipeline/BraTS19_2013_0_1_flair_45.png'
# imgs = io.imread(img_paths).reshape(5, 240, 240).astype('float')
# print(np.max(imgs[0]))
