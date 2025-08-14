# -*- coding: utf-8 -*-
# 脳活動データから中間表現の予測
# (Ridge回帰とNNを用いた予測)/Storage/koba/20200705_Cinet動画視聴fMRIデータ/bold/bold01/DM01

import pickle
import numpy as np
import sys
sys.path.append("../asr/")
from brain_config_u import brainConfig  # 脳活動データの設定ファイル
import sr_data_utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge
import argparse
import tensorflow as tf

frame_num_dict = {}

def createHarray(audio_id, timeseq=False):
    """
    audio_idを与えられて中間表現のnumpy配列を作る
    """
    frame_id_list = brainConfig.createFrameIdList(audio_id)
    if timeseq:
        h_array = np.empty((0, brainConfig.H_DIM_PER_SECOND, brainConfig.H_DIM))
    else:
        h_array = np.empty((0, brainConfig.H_DIM*brainConfig.H_DIM_PER_SECOND), float)
    for frame_id in frame_id_list:
        h_frame_path = "{}{}_h.pkl".format(brainConfig.getHdir(mode='1s', diff=0), frame_id)

        try:
            with open(h_frame_path, "rb") as f:
                if timeseq:
                    h = pickle.load(f)
                    if h.shape[1] != brainConfig.H_DIM_PER_SECOND:
                        print("dim error: {}".format(h_frame_path))
                        continue

                else:
                    h = pickle.load(f)[0].flatten()[np.newaxis, :]
                    if h.shape[1] != brainConfig.H_DIM*brainConfig.H_DIM_PER_SECOND:
                        print("dim error: {}".format(h_frame_path))
                        continue

        except FileNotFoundError:
            print("FileNotFoundError: {}".format(h_frame_path))

        else:
            h_array = np.append(h_array, h, axis=0)

    print("h_array:{}".format(h_array.shape))

    return h_array

def frameShft(brain, timelags):
    """
    timelag秒ずらしたbrainデータを返す
    """
    if len(timelags) == 1:
        frameshift_brain = brain[timelags[0]:]
    else:
        frameshift_brain = np.empty((0, brain.shape[1]*len(timelags)), float)
        for i in range(len(brain)-len(timelags)-timelags[0]):
            concat_brain = brain[i+timelags[0]]
            for t in timelags[1:]:
                concat_brain = np.concatenate([concat_brain, brain[i+t]])
            frameshift_brain = np.append(frameshift_brain, concat_brain[np.newaxis,:], axis=0)

    return frameshift_brain


def loadBrain(audio_id, args):
    """
    脳活動データをロードする
    args:
     - audio_id: string  # 脳活動データの元の音声データのID
     - args: # パラメータ
    returns:
     - brain: np.array  # 脳活動データ
    """
    brain_vset_path = brainConfig.getBrainPath(audio_id, args.mode, args.roi, args.threshold, args.pred_num)
    brain = sr_data_utils.load_pkl(brain_vset_path)

    # 最初の空白の時間を削除(最後は削らない
    brain = brain[brainConfig.BRANK_TIME:]
    brain = frameShft(brain, args.timelag)
    print("done load brain data (brain_id:{}, shape:{})".format(audio_id, brain.shape))
    return brain


def norm(X_train, X_test, mode='mm'):
    """
    正規化
    """

    if mode == 'mm':
        scaler = MinMaxScaler()
    elif mode == 'std':
        scaler = StandardScaler()
    else:
        return X_train, X_test

    if mode == 'mm':
        print("X_train max:{} min:{}".format(X_train.max(), X_train.min()))
        print("X_test max:{} min:{}".format(X_test.max(), X_test.min()))
    elif mode == 'std':
        print("X_train mean:{} std:{}".format(X_train.mean(), X_train.std()))
        print("X_test mean:{} std:{}".format(X_test.mean(), X_test.std()))

    print("start scaler ...")

    # 訓練用のデータを正規化する
    X_train = scaler.fit_transform(X_train)
    # 訓練用データを基準にテストデータも正規化
    X_test = scaler.transform(X_test)

    if mode == 'mm':
        print("X_train max:{} min:{}".format(X_train.max(), X_train.min()))
        print("X_test max:{} min:{}".format(X_test.max(), X_test.min()))
    elif mode == 'std':
        print("X_train mean:{} std:{}".format(X_train.mean(), X_train.std()))
        print("X_test mean:{} std:{}".format(X_test.mean(), X_test.std()))

    return X_train, X_test

def createTrainTest(args, test_mode=False, timeseq=False):
    """
    trainデータとtestデータの作成
    """
    # 脳活動データの次元設定
    brain_dim = brainConfig.getBrainDim(args.mode, args.roi, args.threshold, args.pred_num)

    # データの作成
    X_train = np.empty((0, brain_dim*len(args.timelag)), float)
    X_test = np.empty((0, brain_dim*len(args.timelag)), float)
    if timeseq:
        Y_train = np.empty((0, brainConfig.H_DIM_PER_SECOND, brainConfig.H_DIM), float)
        Y_test = np.empty((0, brainConfig.H_DIM_PER_SECOND, brainConfig.H_DIM), float)
    else:
        Y_train = np.empty((0, brainConfig.H_DIM*brainConfig.H_DIM_PER_SECOND), float)
        Y_test = np.empty((0, brainConfig.H_DIM*brainConfig.H_DIM_PER_SECOND), float)

    # trainデータの作成
    for audio_id in brainConfig.TRAIN_ID:
        print("load {}".format(audio_id))
        h_array = createHarray(audio_id, timeseq=timeseq)
        print("load brain")
        brain = loadBrain(audio_id, args)

        n = h_array.shape[0]
        X_train = np.append(X_train, brain[:n], axis=0)
        Y_train = np.append(Y_train, h_array, axis=0)

        if test_mode:
            break

    print("X_train:{}".format(X_train.shape))
    print("Y_train:{}".format(Y_train.shape))

    # testデータの作成
    for audio_id in brainConfig.TEST_ID:
        print("load {}".format(audio_id))
        h_array = createHarray(audio_id, timeseq=timeseq)
        print("load brain")
        brain = loadBrain(audio_id, args)

        n = h_array.shape[0]
        frame_num_dict[audio_id] = n
        X_test = np.append(X_test, brain[:n], axis=0)
        Y_test = np.append(Y_test, h_array, axis=0)

    print("X_test:{}".format(X_test.shape))
    print("Y_test:{}".format(Y_test.shape))

    # 正規化
    X_train, X_test = norm(X_train, X_test, mode=args.norm)

    return X_train, Y_train, X_test, Y_test

def ridgeRegression(X_train, Y_train, X_test, args):
    """
    ridge回帰を行う
    """
    print("start ridge ...")
    model = Ridge(alpha=args.lamda)
    lr = model.fit(X_train, Y_train)  # 学習
    pred = model.predict(X_test) # 予測
    print("done ridge")

    # 係数と切片を保存する
    # sr_data_utils.write_pkl("w.pkl", lr.coef_)
    # sr_data_utils.write_pkl("b.pkl", lr.intercept_)

    return pred

def nnRegression(X_train, Y_train, X_test, Y_test, args):
    """
    nn回帰を行う
    """
    print("start nn regression ...")

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=str(args.gpu), # specify GPU number
            per_process_gpu_memory_fraction=0.8, # 最大値の50%まで
            allow_growth=True))

    # 変数の定義
    brain_dim = brainConfig.getBrainDim(args.mode, args.roi, args.threshold, args.pred_num)
    x = tf.placeholder("float", [None, brain_dim*len(args.timelag)])
    y_ = tf.placeholder("float", [None, brainConfig.H_DIM*brainConfig.H_DIM_PER_SECOND])

    w_h = tf.Variable(tf.random_normal([brain_dim*len(args.timelag), brainConfig.HIDDEN_DIM],
                                       mean=0.0, stddev=0.05))
    w_o = tf.Variable(tf.random_normal([brainConfig.HIDDEN_DIM, brainConfig.H_DIM*brainConfig.H_DIM_PER_SECOND],
                                       mean=0.0, stddev=0.05))
    b_h = tf.Variable(tf.zeros([brainConfig.HIDDEN_DIM]))
    b_o = tf.Variable(tf.zeros([brainConfig.H_DIM*brainConfig.H_DIM_PER_SECOND]))

    # モデル定義
    h = tf.nn.relu(tf.matmul(x, w_h) + b_h)
    y_hypo = tf.nn.tanh(tf.matmul(h, w_o) + b_o)

    # 学習の設定
    loss = tf.losses.mean_squared_error(y_hypo, y_)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    #train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    accuracy = tf.reduce_mean(tf.square(tf.subtract(y_hypo, y_)))

    loss_list = []
    val_loss_list = []

    # trainning
    init = tf.global_variables_initializer()
    n_batch = brainConfig.batch_size
    n_step = int(len(X_train)/n_batch)

    with tf.Session(config=config) as sess:
        sess.run(init)
        for epoch in range(args.epochs):
            loss_sum = 0

            for i in range(n_step):
                batch_xs = X_train[i*n_batch:(i+1)*n_batch]
                batch_ys = Y_train[i*n_batch:(i+1)*n_batch]
                _, t_loss = sess.run([train_step, loss] ,feed_dict = {x: batch_xs, y_: batch_ys})
                loss_sum += t_loss

            arv_train_loss = loss_sum / n_step #(n_batch * n_step)
            val_loss = sess.run(loss, feed_dict ={x: X_test[:100], y_: Y_test[:100]})
            print('{} epoch / loss {} / val_loss {}'.format(epoch, arv_train_loss, val_loss))
            loss_list.append(arv_train_loss)
            val_loss_list.append(val_loss)

        pred = sess.run(y_hypo, feed_dict = {x: X_test})

    # lossの保存
    loss_dict = {"loss":loss_list, "val_loss":val_loss_list}
    # brain_configにlossのpathを書く
    loss_path = "./NN/loss/{}_{}_{}B_{}_{}_{}_{}.pkl".format(args.rtype,
                                                             args.norm,
                                                             "".join([str(t) for t in args.timelag]),
                                                             args.mode,
                                                             args.roi,
                                                             args.threshold,
                                                             args.pred_num)
    sr_data_utils.write_pkl(loss_path, loss_dict)

    print("done nn regression")

    return pred

def saveResult(pred, args, timeseq=False):
    """
    結果を保存する
    """
    pred_dict = {}
    if not(timeseq):
        pred = pred.reshape(pred.shape[0], brainConfig.H_DIM_PER_SECOND, brainConfig.H_DIM)

    index = 0
    for _id in brainConfig.TEST_ID:
        pred_dict_perAudio = {}
        n = frame_num_dict[_id]
        prediction = pred[index:index+n]
        index = index + n

        for i in range(n):
            pred_dict_perAudio["{}_{:03}".format(_id, i)] = prediction[i]

        pred_dict[_id] = pred_dict_perAudio

    # pickleに保存
    print("save pickle")
    PRED_PATH = "{}{}_{}_{}B_{}_{}_{}_{}.pkl".format(brainConfig.REGRESSION_PATH,
                                                     args.rtype,
                                                     args.norm,
                                                     "".join([str(t) for t in args.timelag]),
                                                     args.mode,
                                                     args.roi,
                                                     args.threshold,
                                                     args.pred_num)
    sr_data_utils.write_pkl(PRED_PATH, pred_dict)

if __name__ == '__main__':
    # 引数の処理
    parser = argparse.ArgumentParser(description='Rigression')
    parser.add_argument('--rtype', '-rt', default=None, type=str,
                        help='ridge or nn')
    parser.add_argument('--norm', '-n', default='std', type=str,
                        help='mm or std or None')
    parser.add_argument('--timelag', '-t', default=None, type=str,
                        help='4,5,6')
    parser.add_argument('--mode', '-m', default=None, type=str,
                        help='roi or corr or pred')
    parser.add_argument('--roi', '-r', default=None, type=str,
                        help='p or t or f or ptf or all')
    parser.add_argument('--threshold', '-tc', default=None, type=int,
                        help='1000 ~ 20000')  # edit1122
    parser.add_argument('--pred_num', '-p', default=None, type=int,
                        help='1000~20000')
    parser.add_argument('--lamda', '-l', default=1, type=int,
                        help='0~10^n')
    parser.add_argument('--gpu', '-g', default=None, type=int,
                        help='0.1.2.3')
    parser.add_argument('--epochs', '-e', default=None, type=int,
                        help='epochs')
    args = parser.parse_args()
    args.timelag = [int(s) for s in args.timelag.split(",")]

    # データ作成
    X_train, Y_train, X_test, Y_test = createTrainTest(args)

    # Y_testの保存
    # saveResult(Y_test, args)

    # 学習&予測
    if args.rtype == "ridge":
        pred = ridgeRegression(X_train, Y_train, X_test, args)
    elif args.rtype == "nn":
        pred = nnRegression(X_train, Y_train, X_test, Y_test, args)

    saveResult(pred, args)