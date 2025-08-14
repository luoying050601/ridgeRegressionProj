import numpy as np
import joblib
import time
import json
from ridge import createDataSet
from utils import correlation_roi,correlation_roi_dict
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
       file_name = 'GloVe_10000000.0.pkl'
    elif  model_type =='word2vec':
        file_name = 'word2vec_1000.0.pkl'
    elif model_type == 'bert-base-uncased':
        file_name = 'bert-base-uncased_50000.0.pkl '
    elif model_type == 'bert-base-multilingual-cased':
        file_name = 'bert-base-multilingual-cased_10000000.0.pkl '
    elif model_type == 'bert-large-uncased-whole-word-masking':
        file_name = 'bert-large-uncased-whole-word-masking_50000.0.pkl'
    elif model_type == 'bert-large-uncased':
         file_name = 'bert-large-uncased_25000.0.pkl'
    elif model_type == 'roberta-large':
        file_name = 'roberta-large_25000.0.pkl'
    elif model_type == 'roberta-base':
        file_name = 'roberta-base_1000.0.pkl'
    elif model_type == 'albert-base-v1':
        file_name = 'albert-base-v1_25000.0.pkl'
    elif model_type == 'albert-large-v1':
        file_name = 'albert-large-v1_100000.0.pkl'
    elif model_type == 'albert-xlarge-v1':
        file_name = 'albert-xlarge-v1_50000.0.pkl'
    elif model_type == 'albert-xxlarge-v1':
        file_name = 'albert-xxlarge-v1_10000.0.pkl'
    elif model_type == 'albert-base-v2':
        file_name = 'albert-base-v2_25000.0.pkl'
    elif model_type == 'albert-large-v2':
        file_name = 'albert-large-v2_100000.0.pkl'
    elif model_type == 'albert-xlarge-v2':
        file_name = 'albert-xlarge-v2_10000000.0.pkl'
    elif model_type == 'albert-xxlarge-v2':
        file_name = 'albert-xxlarge-v2_50000.0.pkl'
    elif model_type == 'gpt2':
        file_name = 'gpt2_10000.0.pkl'
    elif model_type == 'gpt2-medium':
        file_name = 'gpt2-medium_25000.0.pkl'
    elif model_type == 'gpt2-large':
        file_name = 'gpt2-large_25000.0.pkl'
    elif model_type == 'gpt2-xl':
        file_name = 'gpt2-xl_25000.0.pkl'
    elif model_type == 'brainbert':
        file_name = 'brainbert_0.1.pkl'
        # brainbert_10.0.pkl brainbert_100.0.pkl brainbert_10000.0.pkl

    clf = joblib.load(file_name)


    print("test data loading。。。")
    # bert_brain = brainbert_brain 都是相同文本下对应的brain data
    text_before, brain_true = createDataSet('run_', key='test', embedding_model_name=model_list)
    brainbert_text, brainbert_brain = createDataSet('run_', key='test',embedding_model_name='brainbert')
    sc = StandardScaler()
    # X_scaled = sc.fit_transform(X_train)
    text_after = sc.fit_transform(text_before)
    # brainbert_text = sc.fit_transform(brainbert_text)
    # analysis
    score = clf.score(text_after, brain_true)
    # brainbert_score = brainbert_clf.score(brainbert_text, brainbert_brain)
    print("score:", score)
    # print("brainbert_score:", brainbert_score)
    # bert_score: -0.00014841531117118438
    # brai_score: -0.00014208055635837546
    # print("{:.4f}".format(reg.score(X_test, Y_test)))
    brain_pred = clf.predict(text_after)
    # brainbert_pred = brainbert_clf.predict(brainbert_text)
    # corr_list = []

    roi_true, roi_pred = get_roi_data(brain_true, brain_pred)
    corr_dict = correlation_roi_dict(roi_true, roi_pred)
    with open(model_type+"_corr_list_dict.json", 'w') as f:
        json.dump(corr_dict, f)




# brainbert_roi_true,brainbert_roi_pred = get_roi_data(brainbert_brain, brainbert_pred)
# brainbert_corr_dict = correlation_roi_dict(brainbert_roi_true,brainbert_roi_pred)
# with open("brainbert_corr_list_dict.json", 'w') as f:
#     json.dump(brainbert_corr_dict, f)
#
#

# 备份 别删
# brainbert_corr_list = np.array(correlation_roi(brainbert_brain, brainbert_pred))
# print("brainbert_corr_list and average:", brainbert_corr_list.mean())
# with open("brainbert_corr_list.json", 'w') as f:
#     json.dump({"brainbert":brainbert_corr_list.tolist()}, f)

# brainbert_bert_text_corr_list = np.array(correlation_c(brainbert_text, bert_text))
# print("brainbert_bert_text_corr_list and average:", brainbert_bert_text_corr_list.mean())

# brainbert_bert_pred_corr_list = np.array(correlation_c(brainbert_pred, bert_pred))
# print("brainbert_bert_pred_corr_list and average:", brainbert_bert_pred_corr_list.mean())



end = time.perf_counter()
time_cost = ((end - start) / 3600)
print("time-cost(hours):", time_cost)
# 加载
