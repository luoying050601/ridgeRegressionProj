import os
import re
import pandas as pd
import json


imagenet_label_path = '/Storage/ying/project/ridgeRegression/BOLD5000/BOLD5000_Stimuli/Image_Labels/imagenet_final_labels.txt'
il = open(imagenet_label_path,'r')
imagenet_dict = {}
for line in il.readlines():
    line = line.strip()
    k = line.split(' ')[0]
    v = line.split(' ')[1]
    imagenet_dict[k] = v
object = pd.read_pickle(
    r'/Storage/ying/project/ridgeRegression/BOLD5000/BOLD5000_Stimuli/Image_Labels/coco_final_annotations.pkl')
# instances_train2014 = '/Storage/ying/project/ridgeRegression/BOLD5000/BOLD5000_Stimuli/Image_Labels/instances_train2014.json'
captions_train2014 = '/Storage/ying/project/ridgeRegression/BOLD5000/BOLD5000_Stimuli/Image_Labels/captions_train2014.json'
with open(captions_train2014, 'r') as COCO:
    js = json.loads(COCO.read())
    caption_list = js['annotations']
    df = pd.DataFrame(caption_list)
#
# with open(instances_train2014, 'r') as COCO:
#     js = json.loads(COCO.read())
#     cat_list = js['categories']

def find_coco_imagecaption(s):
    label = ''
    # print(s)
    s = s.replace('rep_','')
    # category_id = set()
    index = int(s.split('_')[2])
    df2 = df[df.image_id==index]

    return (max(df2.caption)).lower()

def find_imagenet_imagecaption(s):

    s = s.split('_')[0]
    if s in imagenet_dict:
        label = imagenet_dict[s]
    else:
        label= ''
        print("字典不包含key："+s)


    return label+' '



def func():
    # print(os.sep)  # 显示当前平台下文件路径分隔符
    fileDir = "/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/stim_lists"
    fileList = os.listdir(fileDir)
    coco_index_dict = {}
    imageNetScene_index_dict = {}
    # caption_path = "/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/"+f'*_captioning.txt'
    # if os.path.exists(caption_path):
    #     os.remove(caption_path)

    for file in fileList:
        #
        if "_lists.txt" in file:
            print(file.split('_')[0])
            file_header =file.split('_')[0]
            f1= open(os.path.join(fileDir, file),'rU')

            f2 = open("/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/"+file_header+'_captioning.txt', 'w')
            # is = imageNet+scene
            f3 = open("/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/"+file_header+'_is_captioning.txt', 'w')
            count =0
            coco_index_list=[]
            imageNetScene_index_list=[]
            while True:
                line = f1.readline()
                if not line:
                    break
                line = line.replace('\n', '').replace('.jpg', '').replace('.JPEG', '').replace('.jpe', '')
                # strip('rep_')
                # print(line)  # get image file name
                if "COCO" in line:
                    line = line.replace('rep_', '')
                    label = find_coco_imagecaption(line).replace('\n','')
                    coco_index_list.append(count)
                    f2.write(label)  # 将字符串写入文件中
                    f2.write("\n")  # 换行
                    # print("coco请走这边",line)
                elif "_" in line and (line[0] == 'n' or line.startswith('rep_n')):
                    line = line.replace('rep_', '')
                    imageNetScene_index_list.append(count)

                    label = find_imagenet_imagecaption(line)
                    print("imagenet请走这边",line)
                    f3.write(label)  # 将字符串写入文件中
                    f3.write("\n")  # 换行
                else:
                    # label = str(count)
                    label = re.sub(r'[0-9]+', '', line) + ' '
                    print("scene请走这边", re.sub(r'[0-9]+', '', line))
                    imageNetScene_index_list.append(count)
                    f3.write(label)  # 将字符串写入文件中
                    f3.write("\n")  # 换行


                print(label)
                coco_index_dict[file_header] = coco_index_list
                imageNetScene_index_dict[file_header] = imageNetScene_index_list
                count += 1

            f3.close()
            f2.close()
            f1.close()
            # json_str = json.dumps(coco_index_dict)
            # with open('/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/coco_dict_index.json', 'w') as json_file:
            #     json_file.write(json_str)
            #     json_file.close()
            json_str = json.dumps(imageNetScene_index_dict)
            with open('/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/imageScene_dict_index.json',
                      'w') as json_file:
                json_file.write(json_str)
                json_file.close()            #             +

            print('-------------------------')

                # print(f.readlines())

    # for root, dirs, files in os.walk(fileDir):
        # print('print the absolute path of the directory...')
        # for dir in dirs:
        #     print(os.path.join(root, dir))
        #
        # print('print the absolute path of the file...')
        # for file in files:
        #     print(os.path.join(root, file))
        # print('')


if __name__ == "__main__":
    func()
