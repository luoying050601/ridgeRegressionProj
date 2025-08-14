import numpy as np
import joblib
import time
import json
from ridge import createDataSet
from utils import correlation_roi_dict,correlation_roi_with_pvalue
from sklearn.preprocessing import StandardScaler
roi_index = json.load(open(f'roi_index.json', 'r'))

def get_roi_data(brain_true,brain_pred):
    roi_true = {}
    roi_pred = {}
    for k, v in roi_index.items():
        # k:roi_name v index:0-46840
        if k != 'roi_index':
            region_true = []
            region_pred = []
            for v_ in v:
                region_true.append(brain_true[:,v_])
                region_pred.append(brain_pred[:,v_])

            roi_true[k] = np.array(region_true).T
            roi_pred[k] = np.array(region_pred).T

    return roi_true,roi_pred



start = time.perf_counter()

model_list = [
    'GloVe',
'word2vec',
'bert-base-uncased',
'bert-large-uncased',
'bert-base-multilingual-cased',
'bert-large-uncased-whole-word-masking',
'roberta-large',
'roberta-base',
'albert-base-v1',
'albert-large-v1',
'albert-xlarge-v1',
'albert-xxlarge-v1',
'albert-base-v2',
'albert-large-v2',
'albert-xlarge-v2',
'albert-xxlarge-v2',
'gpt2',
'gpt2-medium',
'gpt2-large',
'gpt2-xl',
    'brainbert'
    ]
#
#
#
#
#
#
for model_type in model_list:
    if model_type =='GloVe':
       file_name = 'is_GloVe_50000.0.pkl'
    if  model_type =='word2vec':
        file_name = 'is_word2vec_25000.0.pkl'
    if model_type == 'bert-base-uncased':
        file_name = 'is_bert-base-uncased_25000.0.pkl'
    if model_type == 'bert-base-multilingual-cased':
        file_name = 'is_bert-base-multilingual-cased_50000.0.pkl'
    if model_type == 'bert-large-uncased-whole-word-masking':
        file_name = 'is_bert-large-uncased-whole-word-masking_25000.0.pkl'
    if model_type == 'bert-large-uncased':
         file_name = 'is_bert-large-uncased_50000.0.pkl'
    if model_type == 'roberta-large':
        file_name = 'is_roberta-large_25000.0.pkl'
    if model_type == 'roberta-base':
        file_name = 'is_roberta-base_50000.0.pkl'
    if model_type == 'albert-base-v1':
        file_name = 'is_albert-base-v1_50000.0.pkl'
    if model_type == 'albert-large-v1':
        file_name = 'is_albert-large-v2_25000.0.pkl'
    if model_type == 'albert-xlarge-v1':
        file_name = 'is_albert-xlarge-v1_10000000.0.pkl'
    if model_type == 'albert-xxlarge-v1':
        file_name = 'is_albert-xxlarge-v1_25000.0.pkl'
    if model_type == 'albert-base-v2':
        file_name = 'is_albert-base-v2_25000.0.pkl'
    if model_type == 'albert-large-v2':
        file_name = 'is_albert-large-v1_50000.0.pkl'
    if model_type == 'albert-xlarge-v2':
        file_name = 'is_albert-xlarge-v2_10000000.0.pkl'
    if model_type == 'albert-xxlarge-v2':
        file_name = 'is_albert-xxlarge-v2_10000000.0.pkl'
    if model_type == 'gpt2':
        file_name = 'is_gpt2_10000000.0.pkl'
    if model_type == 'gpt2-medium':
        file_name = 'is_gpt2-medium_10000000.0.pkl'
    if model_type == 'gpt2-large':
        file_name = 'is_gpt2-large_10000000.0.pkl'
    if model_type == 'gpt2-xl':
        file_name = 'is_gpt2-xl_10000000.0.pkl'
    if model_type == 'brainbert':
        file_name = 'is_brainbert_100000.0.pkl'
        # brainbert_10.0.pkl brainbert_100.0.pkl brainbert_10000.0.pkl

    clf = joblib.load('models/'+file_name)


    print("test data loading。。。")
    print(model_type)
    # bert_brain = brainbert_brain 都是相同文本下对应的brain data
    text_before, brain_true = createDataSet('run_', key='test', embedding_model_name=model_type)
    # brainbert_text, brainbert_brain = createDataSet('run_', key='test',embedding_model_name=model_type)
    sc = StandardScaler()
    # X_scaled = sc.fit_transform(X_train)
    text_after = sc.fit_transform(text_before)
    score = clf.score(text_after, brain_true)
    print("score:", score)
    brain_pred = clf.predict(text_after)
    roi_true, roi_pred = get_roi_data(brain_true, brain_pred)
    corr_dict = correlation_roi_with_pvalue(roi_true, roi_pred)
    with open('pearcorr/coco_'+model_type+"_corr_list_withpvalue.json", 'w') as f:
        json.dump(corr_dict, f)

end = time.perf_counter()
time_cost = ((end - start) / 3600)
print("time-cost(hours):", time_cost)
# 加载
