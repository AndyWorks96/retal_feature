# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/7/18
import nibabel as nib
import matplotlib.pyplot as plt
 
def read_data(path):
    image_data = nib.load(path).get_data()
    return image_data
 
def show_img(ori_img):
    plt.imshow(ori_img[:,:,20], cmap = 'gray') #channel_last
    plt.show()
 
path = 'H:/DataSets/Brats17ValidationData/Brats17_CBICA_AAM_1/Brats17_CBICA_AAM_1_flair.nii.gz'
data = read_data(path)
show_img(data)