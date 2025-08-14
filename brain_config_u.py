# -*- coding: utf-8 -*-
# 脳活動データの設定

class brainConfig():
    # パスの設定
    INTERPOLATE_DIR = "./datasets/interpolate/forESPnet/"
    BRAIN_VSET_DIR = "./datasets/resp/"
    SEGMENT_DIR = "./datasets/seg/"
    REGRESSION_PATH = "./datasets/regression/forESPnet/"
    ROI_PATH = "./datasets/roi/"

    # 脳活動データのIDの設定
    TRAIN_ID = ["S00F0131", "S00M1046", "S00F1396", "S02F1704", "S08M1702",
                "S03F0072", "S04M0497", "S00F0374", "S02F0100", "S02M1715",
                "S04M0790", "S05M1110", "S06M0740", "S01F1707"]
    TEST_ID = ["S02F1109", "S02M1700"]  # ["S04M0497", "S00F0374"]
    ALL_ID = TRAIN_ID + TEST_ID

    # ROI番号の設定
    p_id = [9, 25, 26, 27, 30, 55 + 1, 56 + 1, 65 + 1, 71 + 1]  # 頭頂葉
    t_id = [33, 34, 36, 38, 41, 43 + 1, 72 + 1, 73 + 1, 74 + 1]  # 側頭葉
    f_id = [12, 14, 15, 16, 40, 52 + 1, 53 + 1, 54 + 1]  # 前頭葉

    # matファイルのROI番号の設定(左右含める)
    all_p_id = [200 + i for i in p_id] + [300 + i for i in p_id]
    all_t_id = [200 + i for i in t_id] + [300 + i for i in t_id]
    all_f_id = [200 + i for i in f_id] + [300 + i for i in f_id]
    all_ptf_id = all_p_id + all_t_id + all_f_id

    # 脳活動データの次元の設定(roi)
    DIM = 62552  # 脳活動の全脳皮質の次元
    P_DIM = 8962  # 頭頂葉
    T_DIM = 10245  # 側頭葉
    F_DIM = 11093  # 前頭葉
    PTF_DIM = P_DIM + T_DIM + F_DIM

    # 脳活動データの次元の設定(corr)
    CORR005_DIM = 28980
    CORR010_DIM = 15304
    CORR015_DIM = 7798
    CORR020_DIM = 3983
    CORR025_DIM = 1970
    CORR030_DIM = 926

    # 脳活動データの次元の設定(pred)
    PRED01_DIM = 1000
    PRED05_DIM = 5000
    PRED10_DIM = 10000
    PRED15_DIM = 15000
    PRED20_DIM = 20000

    # 脳活動データの空白の時間
    BRANK_TIME = 10

    # 中間表現の次元
    H_DIM = 1024

    # NNの設定
    HIDDEN_DIM = 10000
    batch_size = 64

    # 中間表現の時系列方向の次元
    H_DIM_PER_SECOND = 25  # 75

    # H_DIM_PER_SECOND_3s = 75

    @classmethod
    def getBrainDim(cls, regression_mode, roi_type, corr_threshold, pred_num):
        # 脳活動データの次元を取得
        if regression_mode == "roi":
            if roi_type == "all":
                brain_dim = cls.DIM
            elif roi_type == "p":
                brain_dim = cls.P_DIM
            elif roi_type == "t":
                brain_dim = cls.T_DIM
            elif roi_type == "f":
                brain_dim = cls.F_DIM
            elif roi_type == "ptf":
                brain_dim = cls.PTF_DIM
        elif regression_mode == "corr":
            brain_dim = corr_threshold  # edit1122
            """
            if corr_threshold == 0.05:
                brain_dim = cls.CORR005_DIM
            elif corr_threshold == 0.1:
                brain_dim = cls.CORR010_DIM
            elif corr_threshold == 0.15:
                brain_dim = cls.CORR015_DIM
            elif corr_threshold == 0.2:
                brain_dim = cls.CORR020_DIM
            elif corr_threshold == 0.25:
                brain_dim = cls.CORR025_DIM
            elif corr_threshold == 0.3:
                brain_dim = cls.CORR030_DIM
            """
        elif regression_mode == "pred":
            if pred_num == 1000:
                brain_dim = cls.PRED01_DIM
            elif pred_num == 5000:
                brain_dim = cls.PRED05_DIM
            elif pred_num == 10000:
                brain_dim = cls.PRED10_DIM
            elif pred_num == 15000:
                brain_dim = cls.PRED15_DIM
            elif pred_num == 20000:
                brain_dim = cls.PRED20_DIM

        return brain_dim

    @classmethod
    def getHdir(cls, mode='IPU', diff=None):
        # H_DIR(中間表現を格納しているディレクトリ)の取得
        if mode == 'IPU':
            H_DIR = "./datasets/h/eval4/"
        elif mode == '1s':
            if diff == 0:
                H_DIR = "./datasets/h/eval5/"
            elif diff == 0.4:
                H_DIR = "./datasets/h/eval6/"
            else:
                print("error: diff of getH_DIR function")
        elif mode == "3s":
            H_DIR = "./datasets/h/eval7/"
        else:
            print("error: mode of getH_DIR function")

        return H_DIR

    @classmethod
    def getBrainPath(cls, audio_id, regression_mode, roi_type, corr_threshold, pred_num):
        # 脳活動データを格納しているパスの取得
        if regression_mode == "roi":
            brain_vset_path = "{}/{}/resp_Listen_{}.pkl".format(cls.BRAIN_VSET_DIR, roi_type, audio_id)
        elif regression_mode == "corr":
            brain_vset_path = "{}/corr_{}/resp_Listen_{}.pkl".format(cls.BRAIN_VSET_DIR, corr_threshold, audio_id)

        elif regression_mode == "pred":
            brain_vset_path = "{}/pred_{}/resp_Listen_{}.pkl".format(cls.BRAIN_VSET_DIR, pred_num, audio_id)

        print(brain_vset_path)

        return brain_vset_path

    @classmethod
    def createFrameIdList(cls, audio_id, mode='1s'):
        # audio_idを与えられて、frame_id(ex. "S00F0131_000")のリストを作る
        # mode: 1s or ipu
        frame_id_list = []
        segment_path = "{}segments_{}".format(cls.SEGMENT_DIR, mode)
        with open(segment_path) as f:
            for s_line in f:
                frame_id, _, _, _ = s_line.split(' ')

                if frame_id.split('_')[0] == audio_id:
                    frame_id_list.append(frame_id)

        return frame_id_list