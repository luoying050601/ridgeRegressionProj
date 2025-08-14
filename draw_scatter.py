import json
# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from utils import getKeyList
# % matplotlib inline

sns.set_style("whitegrid")
sns.set_context("paper")
# 设置风格、尺度

import warnings

warnings.filterwarnings('ignore')


# 不发出警告
def get_data(type, abs_flag):
    # roi = {}
    # roi_index = json.load(open(f'roi_index.json', 'r'))
    roi = json.load(open('pearcorr/' + type + f'_corr_list_withpvalue.json', 'r'))
    # corr_list = corr_list[type]
    # for k, v in roi_index.items():
    #     # k:roi_name v index:0-46840
    #     if k != 'roi_index':
    #         region = []
    #         for v_ in v:
    #             region.append(corr_list[v_])
    #
    #         roi[k] = region
    #
    # with open("roi_json/"+type+"_roi.json", "w") as f:
    #     json.dump(roi, f)
    # f.close()

    df = pd.DataFrame(columns=['brain_roi', 'correlation'])
    average_list = []
    for k, v in roi.items():
        # 插入数据
        if abs_flag:
            v = np.abs(v)
        print(k, np.mean(v))
        average_list.append(np.mean(v))
        for i in range(len(v)):
            df = df.append({"brain_roi": k, "correlation": v[i]}, ignore_index=True)
    if abs_flag:
        filename = 'output/tsv/rr_' + type + '_corr_pabs.tsv'
    else:
        filename = 'output/tsv/rr_' + type + '_corr.tsv'
    df.to_csv(filename, index=False, sep='\t')
    return average_list


def draw_picture(average_line, _type, abs_flag):
    if abs_flag:
        df = pd.read_csv('output/tsv/rr_' + _type + '_corr_pabs.tsv', sep='\t', header=0)
    else:
        df = pd.read_csv('output/tsv/rr_' + _type + '_corr.tsv', sep='\t', header=0)
    df.correlation[df.correlation < 0] = 0
    plt.figure(figsize=(6, 6.5))
    # fig = plt.gcf()
    # fig.set_size_inches(11, 8)
    c = np.array(average_line)
    x = np.array(["Temporal", "Occipital", "Fusiform", "Parietal", "Frontal_Inf", "Thalamus"])
    sns.catplot(x="brain_roi", y="correlation", data=df, kind="box")
    plt.title(_type.upper() + ' Pearson Correlation', fontsize='large', fontweight='bold')  # 设置字体大小与格式
    plt.plot(x, c, color='blue', label='average')
    plt.legend()  # 显示图例
    # fig.add_subplot(111,aspect='equal')
    plt.plot()
    if abs_flag:
        plt.savefig("output/images/" + _type + "_corr_ridge_pabs.png")
    else:
        plt.savefig("output/images/" + _type + "_corr_ridge.png")
    plt.show()


abs_flag = True

model_list = [
    #         'GloVe',
    # 'word2vec',
    # 'bert-base-uncased',
    # 'bert-large-uncased',
    # 'bert-base-multilingual-cased',
    # 'bert-large-uncased-whole-word-masking',
    # 'roberta-large',
    'roberta-base',
    # 'albert-base-v1',
    # 'albert-large-v1',
    # 'albert-xlarge-v1',
    # 'albert-xxlarge-v1',
    # 'albert-base-v2',
    # 'albert-large-v2',
    # 'albert-xlarge-v2',
    # 'albert-xxlarge-v2',
    # 'gpt2',
    # 'gpt2-medium',
    # 'gpt2-large',
    # 'gpt2-xl',
    #         'brainbert'
]
for _type in model_list:
    print(_type)
    average_line = get_data(_type, abs_flag)
    draw_picture(average_line, _type, abs_flag)
# _type = 'roberta'


# _type = 'brainbert'
# print(_type)
# average_line = get_data(_type,abs_flag)
# draw_picture(average_line,_type,abs_flag)
