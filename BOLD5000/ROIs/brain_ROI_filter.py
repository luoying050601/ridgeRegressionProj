import json
import scipy.io as scio
import numpy as np

user = ['CSI1','CSI2','CSI3','CSI4']
roi_area = ['LHPPA','RHPPA',
            'RHLOC','LHLOC',
            'LHEarlyVis','RHEarlyVis',
            "LHRSC",'RHRSC',
            "LHOPA",'RHOPA']

# roi_index = json.load(open(f'/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/coco_dict_index.json', 'r'))
roi_index = json.load(open(f'/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/imageScene_dict_index.json', 'r'))


ROI_data = {}
for file in user:
    # r_array = [][]
    fileDir = "/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/"+file +'/mat/'+file+'_ROIs_TR34.mat'
    # print(fileDir)
    data = scio.loadmat(fileDir)
    roi =  np.concatenate((data['LHPPA'], data['RHPPA'],data['LHOPA'], data['RHOPA'],
                           data['LHEarlyVis'], data['RHEarlyVis'],data['LHRSC'],
                           data['RHRSC'],data['LHLOC'], data['RHLOC']), axis=1)
    ROI_data[file] = roi
    # roi2 =  np.concatenate(PPA,OPA,EarlyVis,RSC,LOC), axis=1)
    # for roi in roi_area:
    #     r_array = r_array + data[roi]
    # print(roi)
# coco_data = {}
is_data = {}
for k,v in ROI_data.items():
    index = roi_index[k]
    # coco_value = []
    is_value = []
    for i in index:
        is_value.append(v[i])
    #   coco_value.append(v[i])
    print(np.array(is_value).shape)
    # coco_data[k] = np.array(coco_value).tolist()
    is_data[k] = np.array(is_value).tolist()
    # 【当前run ImageScene图片数量，当前user的ROI合计】
    # (3119, 1685)
    # (3119, 2270)
    # (3121, 3104)
    # (1834, 2787)
    # 【当前runcoco图片数量，当前user的ROI合计】
    # (2135, 1685)
    # (2135, 2270)
    # (2133, 3104)
    # (1274, 2787)
    # print(coco_data)

# json_str = json.dumps(coco_data)
json_str = json.dumps(is_data)
with open('/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/is_brain_ROI.json', 'w') as json_file:
# with open('/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/coco_brain_ROI.json', 'w') as json_file:
    json_file.write(json_str)
    json_file.close()