import json
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV
import joblib
import time
from create_word_embedding import createDataSet
from utils import make_print_to_file,correlation_roi_with_pvalue
if __name__ == "__main__":
    model_list = [
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
        'GloVe',
        'word2vec',
        'brainbert'

    ]
    import sys
    if len(sys.argv) > 1:
        _alpha = sys.argv[4]
    else:
        _alpha = 10.0

    start = time.perf_counter()
    _type = 'run_'
    make_print_to_file(_type,path='.')
    print("data loading is starting:")
    # データ作成
    for model_type in model_list:
        print(model_type)
        # try:
        X_test, Y_test = createDataSet(_type=_type, key='test', embedding_model_name=model_type)
        X_train, Y_train = createDataSet(_type=_type, key='train', embedding_model_name=model_type)
        a_list = [0.5, 1.0, 5.0, 10.0, 10.0 ** 2, 10.0 ** 3, 10.0 ** 4, 2.5 * (10.0 ** 4), 5.0 * (10.0 ** 4), 10.0 ** 5,
                  10.0 ** 6, 10.0 ** 7]
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X_train)
        X_scaled_t = sc.transform(X_test)
        model = RidgeCV(alphas=a_list)
        model.fit(X_scaled, Y_train)
        print("best alpha_:",model.alpha_)
        # print(model.alpha_)
        print("score:",model.score(X_scaled, Y_train))
        joblib.dump(model, "models/coco_" + model_type + "_" + str(model.alpha_) + ".pkl")
        Y_pre = model.predict(X_scaled_t)
        corr_list = np.array(correlation_roi_with_pvalue(Y_test, Y_pre))
        print(model_type + "Pearson correlation with fdr p-value average:", corr_list.mean())
        with open("pearcorr/is_" + model_type + "_pearcorr_with_pvalue.json", 'w') as f:
            json.dump({model_type: corr_list.tolist()}, f)
        # except Exception as e:
        #     print(e)
        #     pass
        # continue


    end = time.perf_counter()
    time_cost = ((end - start) /3600)
    print("time-cost(hours):", time_cost)
