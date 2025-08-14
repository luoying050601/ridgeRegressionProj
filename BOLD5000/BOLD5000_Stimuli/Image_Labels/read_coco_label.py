

import pickle
import pandas as pd
#
object = pd.read_pickle(r'coco_final_annotations.pkl')
print(object[131967])
# #
# #
# # import json
# # roi_index = json.load(open(f"instances_train2014.json", 'r'))
# # print(roi_index)
# #!/usr/bin/python
#
val_2014 = 'instances_val2014.json'
train_2014 = 'instances_train2014.json'
test_2014 = 'image_info_test2014.json'
# # cat_2017 = './annotations/instances_val2017.json'
#
# import sys, getopt
import json
# from jsonsearch import JsonSearch
#
with open(train_2014, 'r') as COCO:
    js = json.loads(COCO.read())
    print(json.dumps(js['categories']))
#
#     jsondata = JsonSearch(object=js['images'], mode='j')
#     jsondata.search_all_value(key='file_name')
#     print(jsondata)
#
#     # js['annotations'][131967]
#
# # def main(argv):
# #     json_file = None
# #     try:
# #         opts, args = getopt.getopt(argv,"hy:")
# #     except getopt.GetoptError:
# #         # print('.py -t <type>')
# #         sys.exit(2)
# #     for opt, arg in opts:
# #         if opt == '-t':
# #             if arg == 'val':
# #                 json_file = val_2014
# #             elif  arg == 'train':
# #                 json_file = train_2014
# #             else:
# #                 json_file = test_2014
# #     if json_file is not None:
# #         with open(json_file,'r') as COCO:
# #             js = json.loads(COCO.read())
# #             print(json.dumps(js['categories']))
#
# # if __name__ == "__main__":
# #     main(sys.argv[1:])