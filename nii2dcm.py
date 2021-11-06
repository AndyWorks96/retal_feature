# _*_ encoding:utf-8 _*_

import nibabel as nib
import SimpleITK as sitk 

path = 'E:\\BraTS2019_Experiments\\DataSets\\Raw_datas\\MICCAI_BraTS_2019_Data_Training\\LGG\\BraTS19_2013_0_1\\BraTS19_2013_0_1_flair.nii.gz'
# img = nib.load(path)
img = sitk.GetArrayFromImage(sitk.ReadImage(path))
print(img.shape)

save_path = "10002334.dcm"
print(save_path)
# nib.save(img[0, :, :], save_path)
sitk.WriteImage(img[:, :, :], save_path.encode('utf-8'))
# sitk.ImageFileWriter()