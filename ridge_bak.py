import json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler

import seaborn as sns
from sklearn.metrics import \
     r2_score, get_scorer
from transformers import BertModel, BertTokenizer,AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from sklearn.model_selection import \
     cross_validate, train_test_split
from sklearn.linear_model import RidgeCV,Ridge,Lasso
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import time
from create_word_embedding import createDataSet
import lmdb
import msgpack
from lz4.frame import decompress
# from create_word_embedding import get_bert_embedding_tensor,get_brain_bert_attention_output,get_roberta_embedding_tensor
from utils import make_print_to_file,correlation_roi
# from os.path import exists
import torch
#
# # 看一下模型在训练和测试数据的误差表现，以及参数的尺度分布。
# def plot_residuals_and_coeff(resid_train, resid_test, coeff):
#     fig, axes = plt.subplots(1, 3, figsize=(12, 3))
#     axes[0].bar(np.arange(len(resid_train)), resid_train)
#     axes[0].set_xlabel("sample number")
#     axes[0].set_ylabel("residual")
#     axes[0].set_title("training data")
#     axes[1].bar(np.arange(len(resid_test)), resid_test)
#     axes[1].set_xlabel("sample number")
#     axes[1].set_ylabel("residual")
#     axes[1].set_title("testing data")
#     axes[2].bar(np.arange(len(coeff)), coeff)
#     axes[2].set_xlabel("coefficient number")
#     axes[2].set_ylabel("coefficient")
#     fig.tight_layout()
#     return fig, axes
#
# # 创建损失计算函数 SSE
# def sse(resid):
#     return np.sum(resid**2)
# 交差検証
# def cross_validate(train_x_all,train_y_all,a_,split_size=5):
#   results = [0 for _ in range(train_y_all.shape[1])]
#   kf = KFold(n_splits=split_size)
#   for train_idx, val_idx in kf.split(train_x_all, train_y_all):
#     train_x = train_x_all[train_idx]
#     train_y = train_y_all[train_idx]
#     val_x = train_x_all[val_idx]
#     val_y = train_y_all[val_idx]
#
#     reg = Ridge(alpha=a_).fit(train_x,train_y)
#     pre_y = reg.predict(val_x)
#
#     y_val_T = val_y.T
#     y_pre_T = pre_y.T
#     k_fold_r = correlation_c(y_val_T,y_pre_T)
#     results = [x + y for (x, y) in zip(results, k_fold_r)]
#
#   results = map(lambda x : x/5,results)
#   results = list(results)
#   return results
#
# def regmodel_param_plot(
#         validation_score, train_score, alphas_to_try, chosen_alpha,
#         scoring, model_name, test_score=None, filename=None):
#     plt.figure(figsize=(8, 8))
#     sns.lineplot(y=validation_score, x=alphas_to_try,
#                  label='validation_data')
#     sns.lineplot(y=train_score, x=alphas_to_try,
#                  label='training_data')
#     plt.axvline(x=chosen_alpha, linestyle='--')
#     if test_score is not None:
#         sns.lineplot(y=test_score, x=alphas_to_try,
#                      label='test_data')
#     plt.xlabel('alpha_parameter')
#     plt.ylabel(scoring)
#     plt.title(model_name + ' Regularisation')
#     plt.legend()
#     if filename is not None:
#         plt.savefig(str(filename) + ".png")
#     plt.show()
# def regmodel_param_test(
#         alphas_to_try, X, y, cv, scoring='r2',
#         model_name='LASSO', X_test=None, y_test=None,
#         draw_plot=False, filename=None):
#     validation_scores = []
#     train_scores = []
#     results_list = []
#     if X_test is not None:
#         test_scores = []
#         scorer = get_scorer(scoring)
#     else:
#         test_scores = None
#
#     for curr_alpha in alphas_to_try:
#
#         if model_name == 'LASSO':
#             regmodel = Lasso(alpha=curr_alpha)
#         elif model_name == 'Ridge':
#             regmodel = Ridge(alpha=curr_alpha)
#         else:
#             return None
#
#         results = cross_validate(
#             regmodel, X, y, scoring=scoring, cv=cv,
#             return_train_score=True)
#
#         validation_scores.append(np.mean(results['test_score']))
#         train_scores.append(np.mean(results['train_score']))
#         results_list.append(results)
#
#         if X_test is not None:
#             regmodel.fit(X, y)
#             y_pred = regmodel.predict(X_test)
#             test_scores.append(scorer(regmodel, X_test, y_test))
#
#     chosen_alpha_id = np.argmax(validation_scores)
#     chosen_alpha = alphas_to_try[chosen_alpha_id]
#     max_validation_score = np.max(validation_scores)
#     if X_test is not None:
#         test_score_at_chosen_alpha = test_scores[chosen_alpha_id]
#     else:
#         test_score_at_chosen_alpha = None
#
#     if draw_plot:
#         regmodel_param_plot(
#             validation_scores, train_scores, alphas_to_try, chosen_alpha,
#             scoring, model_name, test_scores, filename)
#
#     return chosen_alpha, max_validation_score, test_score_at_chosen_alpha


if __name__ == "__main__":
    model_list=[
        'GloVe',
# 'word2vec',
# 'bert-base-uncased',
# 'bert-large-uncased',
# 'bert-base-multilingual-cased',
# 'bert-large-uncased-whole-word-masking',
# 'roberta-large',
# 'roberta-base',
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
# 'gpt2-xl'
#         'brainbert'
    ]
    import sys
    # bert or brainbert
    # if len(sys.argv)>1:
    #     model_type = sys.argv[2]
    # else:
    #     model_type = 'bert-base-uncased'
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
        try:
            X_test, Y_test = createDataSet(_type=_type,key='test',embedding_model_name=model_type)
            X_train, Y_train = createDataSet(_type=_type,key='train',embedding_model_name=model_type)
            a_list = [0.5, 1.0, 5.0, 10.0, 10.0 ** 2, 10.0 ** 3, 10.0 ** 4, 2.5 * (10.0 ** 4), 5.0 * (10.0 ** 4), 10.0 ** 5,
                      10.0 ** 6, 10.0 ** 7]
            # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            # lasso_alphas = [0.5, 1.0, 5.0, 10.0, 10.0 ** 2, 10.0 ** 3, 10.0 ** 4, 2.5 * (10.0 ** 4), 5.0 * (10.0 ** 4), 10.0 ** 5,
            #           10.0 ** 6, 10.0 ** 7]
            # lasso = Lasso()
            # grid = dict()
            # grid['alpha'] = lasso_alphas
            # model = GridSearchCV(lasso, grid, scoring='neg_mean_absolute_error',
            #     cv=cv, n_jobs=-1)
            sc = StandardScaler()
            X_scaled = sc.fit_transform(X_train)
            X_scaled_t = sc.transform(X_test)
            # results = model.fit(X_scaled, Y_train)
            # print('MAE: %.5f' % results.best_score_)
            # print('Config: %s' % results.best_params_)
            # MAE: -0.00074
            # Config: {'alpha': 0.01}
            # lasso = MultiTaskLassoCV(alphas=lasso_alphas, cv=cv, n_jobs=-1)
            # lasso.fit(X_scaled, Y_train)
            # print('alpha: %.2f' % lasso.alpha_)
            model = RidgeCV(alphas=a_list)
            model.fit(X_scaled, Y_train)
            print(model.alpha_)
            print(model.score(X_scaled_t,Y_test))
            joblib.dump(model, model_type+"_"+str(model.alpha_)+".pkl")

            Y_pre = model.predict(X_scaled_t)
            corr_list = np.array(correlation_roi(Y_test, Y_pre))
            print(model_type+"_corr_list and average:", corr_list.mean())
            with open(model_type+"_corr_list_bak.json", 'w') as f:
                json.dump({model_type: corr_list.tolist()}, f)
        except Exception as e:
            print(e)
            pass
        continue


    end = time.perf_counter()
    time_cost = ((end - start) /3600)
    print("time-cost(hours):", time_cost)
    # 加载
    # estimator = joblib.load('reg_.pkl')

    # sse_test = sse(resid_test)
    # fig, ax = plot_residuals_and_coeff(resid_train, resid_test, model.coef_)
    # 可用空间搜索
    # for n, alpha in enumerate(alphas):
    # model = Ridge(alpha=10.0 ** 5)
    # model.fit(X_train, Y_train)  # 7383 322;7406 321 40 文本少了一组。solved
    # coeffs[n, :] = model.coef_.reshape(-1, )
    # sse_train[n] = sse(Y_train - model.predict(X_train))
    # sse_test[n] = sse(Y_test - model.predict(X_test))

    # resid_train = Y_train - model.predict(X_train)
    # sse_train = sse(resid_train)
    # resid_test = Y_test - model.predict(X_test)
    # sse_test = sse(resid_test)
    # fig, ax = plot_residuals_and_coeff(X_train, Y_train, model.coef_)
    # plt.show()

    # # 绘图
    #
    #
    # # fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    # # for n in range(coeffs.shape[1]):
    # #     axes[0].plot(np.log10(alphas), coeffs[:, n], color='k', lw=0.5)
    # # end = time.perf_counter()
    # # time_cost = ((end - start) / 3600)
    # # axes[1].semilogy(np.log10(alphas), sse_train, label="train")
    # # axes[1].semilogy(np.log10(alphas), sse_test, label="test")
    # # axes[1].legend(loc=0)
    # #
    # # axes[0].set_xlabel(r"${\log_{10}}\alpha$", fontsize=18)
    # # axes[0].set_ylabel(r"coefficients", fontsize=18)
    # # axes[1].set_xlabel(r"${\log_{10}}\alpha$", fontsize=18)
    # # axes[1].set_ylabel(r"sse", fontsize=18)
    # # plt.show()
    # # plt.savefig("coefficients_"+model_type+"_"+str(time.time())+".png")
    # # ridge cv
    # # ridge_cv = RidgeCV(alphas=alphas,cv=5)
    # # reg = ridge_cv.fit(X_train, Y_train)
    # # score_train = reg.score(X_train,Y_train)

    # # pred = reg.predict(X_test)
    # # corr_list = []
    # # corr_list = np.array(correlation_c(Y_test, pred))

    # #
    # X_train = standardization(X_train, 'train',mode="std")
    #     # Y_train = standardization(Y_train, 'train',mode="std")
    #     # X_test = standardization(X_test, 'test',mode="std")
    #     # Y_test = standardization(Y_test, 'test',mode="std")
    #     # end = time.perf_counter()
    #     # time_cost = ((end - start) / 3600)
    #     # n_alphas = 20
    #     # alphas count is 200, 都在10的-x次方之间
    #     # alphas = np.logspace(0, 3, n_alphas)
    #     # alphas = [0.5, 1.0, 5.0, 10.0, 10.0 ** 2, 10.0 ** 3, 10.0 ** 4, 2.5 * (10.0 ** 4), 5.0 * (10.0 ** 4), 10.0 ** 5,
    #     #           10.0 ** 6, 10.0 ** 7]
    #     # alphas = [ 10.0 ** 5 ]
    #
    #     # coeffs = np.zeros((len(alphas), X_train.shape[1]*Y_train.shape[1]))
    #     # sse_train = np.zeros_like(alphas)
    #     # sse_test = np.zeros_like(alphas)
    #
    #     # model = MultiTaskLassoCV()
    #     # model.fit(X_train, Y_train)
    #     # model = RidgeCV()