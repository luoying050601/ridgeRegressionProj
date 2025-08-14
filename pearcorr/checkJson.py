import json
import nibabel as nib
import numpy as np


roi_index = json.load(open(f'/Storage/ying/project/ridgeRegression/brainbert_roi.json', 'r'))
print(roi_index)


corr_list = json.load(open(f'brainbert_corr_list_withpvalue.json', 'r'))
print(corr_list)


#/Storage/ying/resources/BOLD5000/derivatives/fmriprep/sub-CSI1/anat/sub-CSI1_T1w_preproc.nii.gz
img = nib.load('/Storage/ying/resources/BOLD5000/derivatives/fmriprep/sub-CSI1/anat/sub-CSI1_T1w_label-aparcaseg_roi.nii.gz')
# img_affine = img.affine
img = img.get_data()
x = np.bincount(img.flatten())
print(x)
print(img.shape)
#
# coding:UTF-8
import scipy.io as scio

dataFile = '/Storage/kanenko/OG/Shimako/peano/Handover_from_Ozaki/original_data/TV/vset/YO.vset/vset_099.mat'
data = scio.loadmat(dataFile)
print(data)


