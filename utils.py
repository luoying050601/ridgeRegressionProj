#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.stats as st
# from multipy.fdr import qvalue
import statsmodels.stats.multitest as sm
import sys
import os
import os.path
import importlib.machinery as imm
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import math
import numpy as np
import numpy.matlib
from scipy import stats, linalg
import h5py
import ctypes

import matplotlib
import matplotlib.pyplot as plt

matplotlib.pyplot.switch_backend("Agg")

util_logger = logging.getLogger(os.path.basename(__file__))
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
LOG_DIR = Proj_dir + '/output/'


def normalization(data):
    import numpy as np
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    # normData = np.zeros(np.shape(data))
    # m = normData.shape[0]
    normData = data - np.tile(minVals, np.shape(data))
    normData = normData / np.tile(ranges, np.shape(data))
    return normData, ranges, minVals


def standardization(data, _type, mode='mm'):
    """
    正規化
    """

    global scaler
    if mode == 'mm':
        scaler = MinMaxScaler()
    elif mode == 'std':
        scaler = StandardScaler()
    # else:
    #     return data

    if mode == 'mm':
        print("before standardization, data max:{} min:{}".format(data.max(), data.min()))
        # print("X_test max:{} min:{}".format(X_test.max(), X_test.min()))
    elif mode == 'std':
        print("before standardization, data mean:{} std:{}".format(data.mean(), data.std()))
        # print("X_test mean:{} std:{}".format(X_test.mean(), X_test.std()))

    print("start scaler ...")

    # 訓練用のデータを正規化する
    if _type == 'train':
        data = scaler.fit_transform(data)
    else:
        scaler.fit(data)
        # 訓練用データを基準にテストデータも正規化
        data = scaler.transform(data)

    if mode == 'mm':
        print("after standardization, data max:{} min:{}".format(data.max(), data.min()))
        # print("X_test max:{} min:{}".format(X_test.max(), X_test.min()))
    elif mode == 'std':
        print("after standardization, data mean:{} std:{}".format(data.mean(), data.std()))
        # print("X_test mean:{} std:{}".format(X_test.mean(), X_test.std()))

    return data


def make_print_to_file(_type, path=LOG_DIR):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import os
    # import config_file as cfg_file
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime(_type + '%Y-%m-%d-%H:%M:%S')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # print -> log
    #############################################################
    print(fileName.center(60, '*'))


def correlation_roi_with_pvalue(y_true, y_pre):
    corr_roi_list = []
    # roi_corr_dict = {}
    for index, (k_t, k_p) in enumerate(zip(y_true, y_pre)):
        corr, p = st.pearsonr(k_t, k_p)
        if p >= 0.05:
            corr = 0
        corr_roi_list.append(corr)
    return corr_roi_list

    # for j in range(y_true.shape[0]):
    #     import scipy.stats as st
    #     corr, p = st.pearsonr(y_true[j, :],y_pre[j, :])
    #     if p >= 0.05:
    #         corr = 0
    #     corr_list.append(corr)
    # return corr_list


def correlation_roi_dict(y_true, y_pre):
    roi_corr_dict = {}
    for (k_t, k_p) in zip(y_true, y_pre):
        v_t = y_true[k_t]
        v_p = y_pre[k_p]
        corr_roi_list = []
        p_list = []
        for i in range(v_t.shape[1]):
            corr, p = st.pearsonr(v_t[:, i], v_p[:, i])
            corr_roi_list.append(corr)
            p_list.append(p)
        # _, qvals = qvalue(p_list)
        a = sm.multipletests(p_list, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
        # qvals
        for i in range(len(a[1])):
            if a[1][i] >= 0.05:
                corr_roi_list[i] = 0
        print(k_t, np.array(corr_roi_list).mean())
        roi_corr_dict[k_t] = corr_roi_list

    # ave_corr = np.average(corr_list)
    # print("相関係数: {}".format(corr_list))

    return roi_corr_dict


def getKeyList(dict):
    return [*dict]


def correlation_roi(y_true, y_pre):
    import scipy.stats as st
    corr_roi_list = []
    # corr_roi_list = np.zeros_like(y_true, dtype=np.int)
    for i in range(y_true.shape[1]):
        corr, p = st.pearsonr(y_true[:, i], y_pre[:, i])
        # print("corr:", corr)
        # print("p value:", p)
        # st.peasonerの返り値は(相関係数, p値)
        if p >= 0.05:
            corr = 0
        # corr = np.corrcoef(y_test_T[j, :],y_pre_T[j, :])[0,1]
        # corr_roi_list[j,i] = corr

        corr_roi_list.append(corr)
    # ave_corr = np.average(corr_list)
    # print("相関係数: {}".format(corr_list))

    return corr_roi_list


###
# Get experimental information
###
def get_expinfo(param):
    fname = param["expInfoFile"]
    if os.path.isfile(fname):
        ei_module = imm.SourceFileLoader("expinfo", fname).load_module()
        ei, infovoxel, vsetform = ei_module.main(param)
    else:
        raise Exception("Not found experimental information: {0}".format(fname))

    return ei, infovoxel, vsetform


###
# Get resampling indices for cross validation
###
def get_resamp_ind(samplenum, divsample=None, ncvsets=10, seed=1234):
    if not divsample:
        divnum = 50
    else:
        divnum = int(samplenum / divsample)

    # Get cross validation samples
    np.random.seed(seed)

    xs = math.floor(samplenum / divnum)
    xr = np.reshape(np.arange(xs * divnum), (xs, divnum), order="F")

    ssinds = []
    for _ in range(ncvsets):
        a = np.random.permutation(divnum)
        reg_ind = xr[:, a[range(round(divnum / ncvsets))]]
        reg_ind = np.ravel(reg_ind, order="F").T
        trn_ind = np.setdiff1d(np.arange(samplenum), reg_ind)
        ssinds.append({"regInd": reg_ind, "trnInd": trn_ind})

    return ssinds


###
# Load fMRI response data for model training
###
def load_fmri_trndata(expinfo, fnorm=False, logger=util_logger):
    if "trnconcatresp" in expinfo:

        data = []
        for ii, resp_file in enumerate(expinfo["trnconcatresp"]):
            fname = resp_file
            logger.info("Loading {0}...".format(fname))

            with h5py.File(fname, "r") as f:
                t = f["data"][()]
            root, ext = os.path.splitext(fname)
            if ext == ".mat":
                t = t.T

            sdt = t.shape
            t = np.reshape(t, sdt)

            if fnorm:
                logger.info("Normalizing...")
                t = np.transpose(norm_std_mean(t.T))

            if ii == 0:
                data = np.empty(sdt)

            data = np.concatenate((data, t), axis=1)

    else:

        vframes = expinfo["trnframes"]
        runnum = len(expinfo["trnresp"])
        datasize = expinfo["datasize"]

        # Load voxel responses
        framenums = []
        framenum = None

        if type(vframes) == list:
            framenums = list(map(lambda x: len(x), vframes))
            data = np.zeros((np.prod(datasize), sum(framenums)), dtype=np.float32)
        else:
            framenum = len(vframes)
            data = np.zeros((np.prod(datasize), framenum * runnum), dtype=np.float32)

        for ii in range(runnum):
            fname = expinfo["trnresp"][ii]
            logger.info("Loading {0}...".format(fname))
            with h5py.File(fname, "r") as f:  # load dataDT
                data_dt = f["dataDT"][()].T

            sdt = data_dt.shape
            data_dt = np.reshape(data_dt, (-1, sdt[-1]), order="F")

            if fnorm:
                logger.info("Normalizing...")
                data_dt, __, __ = norm_std_mean(data_dt.T)
                data_dt = data_dt.T

            if vframes is not None:
                if type(vframes) == list:
                    data[:, np.arange(framenums[ii]) + sum(framenums[:ii])] = data_dt[:, vframes[ii]]
                else:
                    data[:, np.arange(framenum) + (ii - 1) * framenum] = data_dt[:, vframes]
            else:
                framenum = data_dt.shape[1]
                data[:, np.arange(framenum) + (ii - 1) * framenum] = data_dt

    return data.T


###
# Load fMRI response data for model validation
###
def load_fmri_valdata(expinfo, fnorm=False, fraw=None, logger=util_logger):
    tdata = []
    tdata_mean = []
    for di, vr in enumerate(expinfo["valresp"]):
        # Load the stimulus sequence
        seqfile = vr["seq"]
        if not seqfile:
            logger.info("Stimulus sequence file not found.")
        else:
            sys.exit()

        # Load voxel signals
        if "valconcatresp" in vr:

            fname = vr["valconcatresp"]
            logger.info("Loading {0}...".format(fname))
            with h5py.File(fname, "r") as f:
                data = f["data"][()]
            root, ext = os.path.splitext(fname)
            if ext == ".mat":
                data = data.T

            sdt = data.shape
            data = np.reshape(data, sdt)
            if fnorm:
                logger.info("Normalizing...")
                data = np.transpose(norm_std_mean(data.T))

            data_mean = data.T
            data = []

        else:

            data = []
            for ii, fname in enumerate(vr["resp"]):

                if not fraw:
                    logger.info("Loading {0}...".format(fname))
                    with h5py.File(fname, "r") as f:
                        data_dt = f["dataDT"][()]
                        root, ext = os.path.splitext(fname)
                        if ext == ".mat":
                            data_dt = data_dt.T
                else:
                    fname = fraw[di]["fname"]
                    blockname = fraw[di]["blockname"][ii]
                    logger.info("Loading {0} : {1}...".format(fname, blockname))
                    with h5py.File(fname, "r") as f:
                        data_dt = f[blockname][()].astype(np.float32)

                sdt = data_dt.shape
                data_dt = np.reshape(data_dt, sdt)
                if fnorm:
                    logger.info("Normalizing...")
                    data_dt, __, __ = norm_std_mean(data_dt.T)
                    data_dt = data_dt.T

                if expinfo["valframes"]:
                    data_dt = data_dt[:, expinfo["valframes"]]

                if ii == 0:
                    data = data_dt
                else:
                    data = np.concatenate((data, data_dt), axis=1)

            logger.info("Re-aligning data...")
            voxelnum = np.prod(expinfo["datasize"])
            data = np.reshape(data, (voxelnum, int(np.ceil(expinfo["valclipsec"] / expinfo["TRsec"])),
                                     expinfo["valrepnum"], -1), order="F")
            data = data.transpose((0, 1, 3, 2))
            data = np.reshape(data, (voxelnum, -1, expinfo["valrepnum"]), order="F")
            data_mean = np.squeeze(np.nanmean(data, axis=2)).T
            data = data.transpose((1, 2, 0))

        if di == 0:
            tdata = data
            tdata_mean = data_mean
        else:
            tdata = np.concatenate((tdata, data), axis=0)
            tdata_mean = np.concatenate((tdata_mean, data_mean), axis=0)

    return tdata_mean, tdata


###
# Make a regressor matrix with hemodynamic delay terms
###
def make_delay_matrix(x, delays, block_size=None, in_pad_width=None, out_pad_width=None):
    n_samps = x.shape[0]
    if block_size is None:
        # If block_size is not defined
        ranges = [[0, n_samps], ]
    elif isinstance(block_size, list):
        # If block_size is list
        ranges = ([[sum(block_size[:i]), sum(block_size[:(i + 1)])] for i in range(0, len(block_size))])
    else:
        # If block_size is single value
        ranges = [[x, x + block_size] for x in range(0, n_samps, block_size)]

    # Calculate padding width from delays
    delay_array = np.array(delays)
    pos_delays = delay_array[delay_array > 0]
    neg_delays = delay_array[delay_array < 0]
    pad_before = np.max(pos_delays) if pos_delays.size > 0 else 0
    pad_after = np.absolute(np.min(neg_delays)) if neg_delays.size > 0 else 0
    delay_pad_width = (pad_before, pad_after)

    not_pad = False
    if in_pad_width is None and out_pad_width is None:
        not_pad = True
    elif in_pad_width is None:
        in_pad_width = (0, 0)
    elif out_pad_width is None:
        out_pad_width = (0, 0)

    if not_pad:
        add_pad_width = (0, 0)
        remove_pad_width = (0, 0)
    else:
        # Calulate padding width to add
        add_pad_before = np.clip(delay_pad_width[0] - in_pad_width[0] + out_pad_width[0], a_min=0, a_max=None)
        add_pad_after = np.clip(delay_pad_width[1] - in_pad_width[1] + out_pad_width[1], a_min=0, a_max=None)
        add_pad_width = (add_pad_before, add_pad_after)

        # Calulate padding width to remove
        remove_pad_before = in_pad_width[0] + add_pad_width[0] - out_pad_width[0]
        remove_pad_after = in_pad_width[1] + add_pad_width[1] - out_pad_width[1]
        remove_pad_width = (remove_pad_before, remove_pad_after)

    samp_st = 0
    samp_ed = 0
    for i, r in enumerate(ranges):
        # Extract frames in the block
        x_blk = x[r[0]:r[1], :]

        # Pads block matrix
        padded_x_blk = np.pad(x_blk, [add_pad_width, (0, 0)], 'edge')
        n_padded_samps = padded_x_blk.shape[0]

        # Concatenate over delays
        delayed_x_blk_list = [np.roll(padded_x_blk, d, axis=0) for d in delays]
        delayed_x_blk = np.concatenate(delayed_x_blk_list, axis=1)

        # Remove padding
        delayed_x_blk = delayed_x_blk[remove_pad_width[0]:n_padded_samps - remove_pad_width[1], :]

        # Concatenate over blocks
        if i == 0:
            n_delayed_samps = n_samps + len(ranges) * (sum(add_pad_width) - sum(remove_pad_width))
            y = np.empty((n_delayed_samps, delayed_x_blk.shape[1]))

        samp_ed = samp_st + delayed_x_blk.shape[0]
        y[samp_st:samp_ed, :] = delayed_x_blk
        samp_st = samp_ed

    return y


###
# Normalization
###
def norm_std_mean(s, s_stds=None, s_means=None):
    divnum = 20
    snum = s.shape[1]
    chunksize = math.ceil(snum / divnum)

    if s_means is None:
        s_means = np.mean(s, axis=0)

    if s_stds is None:
        # Calculate standard
        if s.size < 100000 or s.shape[1] < 20:
            s_stds = np.std(s, ddof=1, axis=0)
        else:
            s_stds = np.empty(snum, dtype=np.float32)
            st = 0
            for _ in range(divnum):
                ed = min([snum, st + chunksize])
                s_stds[st:ed] = np.std((s[:, st:ed]), ddof=1, axis=0)
                st = ed

    s_norm = np.empty(s.shape)

    # Normalization
    s_stds[s_stds == 0] = 1  # avoid zero-division

    if s.size < 100000 or s.shape[1] < 20:
        s_norm = s - s_means
        s_norm /= s_stds
    else:
        st = 0
        for _ in range(divnum):
            ed = min([snum, st + chunksize])
            s_norm[:, st:ed] = (s[:, st:ed] - s_means[st:ed]) / s_stds[st:ed]
            st = ed

    return s_norm, s_stds, s_means


###
# Pairwise correlation with p values
###
def pair_corr_with_p(s1, s2):
    cc = pair_corr(s1, s2)

    n = s1.shape[0]
    t = np.dot(cc, np.sqrt(n - 2)) / np.sqrt(1 - np.power(cc, 2))
    p = (1 - stats.t.cdf(np.absolute(t), n - 2)) * 2

    return cc, p


###
# Pairwise correlation
###
def pair_corr(s1, s2):
    s1_norm, __, __ = norm_std_mean(s1)
    s2_norm, __, __ = norm_std_mean(s2)

    cc = np.sum((s1_norm * s2_norm), axis=0) / (s1_norm.shape[0] - 1)

    return cc


###
# PCA for small data samples with high dimensionality
###
def pca_alter(x):
    x_center = x - np.matlib.repmat(np.mean(x, axis=0), x.shape[0], 1)

    s, v, d = linalg.svd(np.dot(x_center, x_center.transpose()), lapack_driver='gesvd')

    coeff1 = np.dot(x_center.transpose(), s) / np.matlib.repmat(np.sqrt(v), x.shape[1], 1)
    score1 = np.dot(x_center, coeff1)
    latent1 = np.diag(np.cov(score1.transpose()))

    return coeff1, score1, latent1


###
# Z-score function
###
def zs(v): return (v - v.mean(0)) / v.std(0)


###
# Multiply a matrix by a diagonal matrix
###
def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))

    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d * mtx.T).T
    else:
        return d * mtx


###
# Singular value decomposition
###
def _real_type(t, default):
    _real_types_map = {np.single: np.single,
                       np.double: np.double,
                       np.csingle: np.single,
                       np.cdouble: np.double}
    return _real_types_map.get(t, default)


def _complex_type(t, default):
    _complex_types_map = {np.single: np.csingle,
                          np.double: np.cdouble,
                          np.csingle: np.csingle,
                          np.cdouble: np.cdouble}
    return _complex_types_map.get(t, default)


def _common_type(*arrays):
    # in lite version, use higher precision (always double or cdouble)
    result_type = np.single
    is_complex = False
    for a in arrays:
        if issubclass(a.dtype.type, np.inexact):
            if issubclass(a.dtype.type, np.complexfloating):
                is_complex = True
            rt = _real_type(a.dtype.type, None)
            if rt is None:
                # unsupported inexact scalar
                raise TypeError("array type %s is unsupported in linalg" %
                                (a.dtype.name,))
        else:
            rt = np.double
        if rt is np.double:
            result_type = np.double
    if is_complex:
        t = np.cdouble
        result_type = _complex_type(result_type, np.cdouble)
    else:
        t = np.double
    return t, result_type


def _fast_copy_and_transpose(t, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.type is t:
            cast_arrays = cast_arrays + (np.fastCopyAndTranspose(a),)
        else:
            cast_arrays = cast_arrays + (np.fastCopyAndTranspose(a.astype(type)),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays


def svd_dgesvd(a, full_matrices=1, compute_uv=1):
    """
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices, ``U`` and ``Vh``,
    and a 1-dimensional array of singular values, ``s`` (real, non-negative),
    such that ``a == U S Vh``, where ``S`` is the diagonal
    matrix ``np.diag(s)``.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to decompose
    full_matrices : boolean, optional
        If True (default), ``U`` and ``Vh`` are shaped
        ``(M,M)`` and ``(N,N)``.  Otherwise, the shapes are
        ``(M,K)`` and ``(K,N)``, where ``K = min(M,N)``.
    compute_uv : boolean
        Whether to compute ``U`` and ``Vh`` in addition to ``s``.
        True by default.

    Returns
    -------
    U : ndarray, shape (M, M) or (M, K) depending on `full_matrices`
        Unitary matrix.
    s :  ndarray, shape (K,) where ``K = min(M, N)``
        The singular values, sorted so that ``s[i] >= s[i+1]``.
    Vh : ndarray, shape (N,N) or (K,N) depending on `full_matrices`
        Unitary matrix.

    Raises
    ------
    LinAlgError
        If SVD computation fails.
        For details see dgesvd.f and dbdsqr.f of LAPACK
    """

    # Make an array
    a = np.asarray(a)
    wrap = getattr(a, "__array_wrap__", a.__array_wrap__)

    # Assert the dimensionality of array items
    for x in a:
        if len(x.shape) != 2:
            raise np.linalg.LinAlgError("{}-dimensional array given. Array must be \
                two-dimensional".format(len(x.shape)))

    # Assert non empty
    for x in a:
        if np.size(x) == 0:
            raise np.linalg.LinAlgError("Arrays cannot be empty")

    m, n = a.shape
    t, result_t = _common_type(a)
    real_t = np.double
    a = _fast_copy_and_transpose(t, a)
    s = np.zeros((min(n, m),), real_t)

    if compute_uv:
        if full_matrices:
            nu = m
            nvt = n
            option = 'A'
        else:
            nu = min(n, m)
            nvt = min(n, m)
            option = 'S'
        u = np.zeros((nu, m), t)
        vt = np.zeros((n, nvt), t)
    else:
        option = 'N'
        nvt = 1
        u = np.empty((1, 1), t)
        vt = np.empty((1, 1), t)

    lib = ctypes.CDLL('/usr/lib64/liblapack.so')
    lapack_routine = lib.dgesvd_

    lwork = 1
    work = np.zeros((lwork,), t)
    info = ctypes.c_int(0)
    m = ctypes.c_int(m)
    n = ctypes.c_int(n)
    nvt = ctypes.c_int(nvt)
    lwork = ctypes.c_int(-1)
    lapack_routine(option, option, m, n, a, m, s, u, m, vt, nvt,
                   work, lwork, info)
    if info.value < 0:
        raise Exception('%d-th argument had an illegal value' % info.value)

    lwork = int(work[0])
    work = np.zeros((lwork,), t)
    lwork = ctypes.c_int(lwork)
    lapack_routine(option, option, m, n, a, m, s, u, m, vt, nvt,
                   work, lwork, info)
    if info.value > 0:
        raise Exception('Error during factorization: %d' % info.value)
    #        raise LinAlgError, 'SVD did not converge'
    s = s.astype(_real_type(result_t, np.double))
    if compute_uv:
        u = u.transpose().astype(result_t)
        vt = vt.transpose().astype(result_t)
        return wrap(u), s, wrap(vt)
    else:
        return s


###
# Ridge regression
###
def ridge(stim, resp, alpha, normalpha=False, logger=util_logger):
    """Uses ridge regression to find a linear transformation of [stim] that approximates
    [resp]. The regularization parameter is [alpha].

    Parameters
    ----------
    stim : array_like, shape (T, N)
        Stimuli with T time points and N features.
    resp : array_like, shape (T, M)
        Responses with T time points and M separate responses.
    alpha : float or array_like, shape (M,)
        Regularization parameter. Can be given as a single value (which is applied to
        all M responses) or separate values for each response.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value of stim. Good for
        comparing models with different numbers of parameters.
    logger : Logger object

    Returns
    -------
    wt : array_like, shape (N, M)
        Linear regression weights.
    """
    try:
        u, s, vh = linalg.svd(stim, full_matrices=False)
    except linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        u, s, vh = svd_dgesvd(stim, full_matrices=False)

    ur = np.dot(u.T, np.nan_to_num(resp))

    # Expand alpha to a collection if it's just a single value
    if isinstance(alpha, float):
        alpha = np.ones(resp.shape[1]) * alpha

    # Normalize alpha by the LSV norm
    norm = s[0]
    if normalpha:
        nalphas = alpha * norm
    else:
        nalphas = alpha

    # Compute weights for each alpha
    ualphas = np.unique(nalphas)
    wt = np.zeros((stim.shape[1], resp.shape[1]))
    for ua in ualphas:
        selvox = np.nonzero(nalphas == ua)[0]
        # awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
        awt = vh.T.dot(np.diag(s / (s ** 2 + ua ** 2))).dot(ur[:, selvox])
        wt[:, selvox] = awt

    return wt


###
# Ridge regression to determine the best regularization coefficient
###
def ridge_corr(rstim, pstim, rresp, presp, alphas, normalpha=False, corrmin=0.2,
               singcutoff=1e-10, use_corr=True, logger=util_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT returned, because
    computing the correlations without computing regression weights is much, MUCH faster.

    Parameters
    ----------
    rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    logger : Logger object

    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.

    """
    # Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        u, s, vh = linalg.svd(rstim, full_matrices=False)
    except linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        u, s, vh = svd_dgesvd(rstim, full_matrices=False)

    # Truncate tiny singular values for speed
    origsize = s.shape[0]
    ngood_s = np.sum(s > singcutoff)
    nbad = origsize - ngood_s
    u = u[:, :ngood_s]
    s = s[:ngood_s]
    vh = vh[:ngood_s]
    logger.info("Dropped %d tiny singular values.. (U is now %s)" % (nbad, str(u.shape)))

    # Normalize alpha by the LSV norm
    norm = s[0]
    logger.info("Training stimulus has LSV norm: %0.03f" % norm)
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    # Precompute some products for speed
    ur = np.dot(u.T, rresp)  # Precompute this matrix product for speed
    pvh = np.dot(pstim, vh.T)  # Precompute this matrix product for speed

    # Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    z_presp = zs(presp)
    # Prespvar = Presp.var(0)
    prespvar_actual = presp.var(0)
    prespvar = (np.ones_like(prespvar_actual) + prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (prespvar_actual - prespvar).mean())
    rcorrs = []  # Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        # D = np.diag(S/(S**2+a**2)) # Reweight singular vectors by the ridge parameter
        d = s / (s ** 2 + na ** 2)  # Reweight singular vectors by the (normalized?) ridge parameter

        pred = np.dot(mult_diag(d, pvh, left=False), ur)  # Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better (2.0 seconds to prediction in test)

        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) # Pretty good (2.4 seconds to prediction in test)
        # pred = np.dot(pvhd, UR)

        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) # Bad (14.2 seconds to prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) # Worst
        # pred = np.dot(Pstim, wt) # Predict test responses

        if use_corr:
            # prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) # Compute predicted test response norms
            # Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1]
            #                   for ii in range(Presp.shape[1])]) # Slowly compute correlations
            # Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()\
            #         /(prednorms*Prespnorms) # Efficiently compute correlations
            rcorr = (z_presp * zs(pred)).mean(0)
        else:
            # Compute variance explained
            resvar = (presp - pred).var(0)
            rsq = 1 - (resvar / prespvar)
            rcorr = np.sqrt(np.abs(rsq)) * np.sign(rsq)

        rcorr[np.isnan(rcorr)] = 0
        rcorrs.append(rcorr)

        log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        log_msg = log_template % (a,
                                  np.mean(rcorr),
                                  np.max(rcorr),
                                  corrmin,
                                  np.sum(rcorr > corrmin) - np.sum(-rcorr > corrmin))
        logger.info(log_msg)

    return rcorrs


###
# Model construction with ridge regression
###
def train_ridge(x, y, ssinds, lambdas, rep_cv=None, logger=util_logger, separate=False):
    n_lambdas = len(lambdas)

    if not rep_cv:
        rep_cv = len(ssinds)

    cv_ccs = np.zeros((y.shape[1], n_lambdas), dtype=np.float32)

    for vi in range(rep_cv):
        logger.info("Cross validation {0}/{1}:".format(vi + 1, rep_cv))
        ccs = ridge_corr(x[ssinds[vi]["trnInd"], :], x[ssinds[vi]["regInd"], :],
                         y[ssinds[vi]["trnInd"], :], y[ssinds[vi]["regInd"], :],
                         lambdas, logger=logger)
        cv_ccs += np.transpose(ccs)  # Transposed with conversion from list to numpy array

    # Average over repetitions
    cv_ccs = cv_ccs / rep_cv

    # Mean ccs for each lambda
    mean_ccs = np.nanmean(cv_ccs, axis=0)

    # Get the best lambda and the corresponding cc
    bmccs = np.max(mean_ccs)
    bind = np.argmax(mean_ccs)
    blambda = lambdas[bind]
    logger.info("Uniform regularization: best lambda={0:.0f}, mean ccs={1:.5f}".format(blambda, bmccs))

    param = dict()

    if separate:
        # Separate regularization

        best_ccs = np.max(cv_ccs, axis=1)
        best_ids = np.argmax(cv_ccs, axis=1)

        mean_best_ccs = np.nanmean(best_ccs)
        logger.info("Separate regularization: mean_ccs={0:.5f}".format(mean_best_ccs))

        blambda = lambdas[best_ids]

        # Store results
        param["bestLambdas"] = blambda
        param["bestCcs"] = best_ccs
        param["sepCcs"] = cv_ccs

    # Estimate weights using the whole samples
    logger.info("Estimating weights using the entire samples...")
    ws = ridge(x, y, blambda, logger=logger)

    # Store results
    param["lambdas"] = lambdas
    param["meanCcs"] = mean_ccs
    param["ccs"] = cv_ccs[:, bind]

    return param, ws.astype("float32")


###
# Initialize figure for file output
###
def initfigure(orientation=None, fontsize=None, linewidth=None):
    if orientation == "vertical":
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 vertical
    elif orientation == "horizontal":
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 horizontal
    else:
        fig = plt.figure()

    if fontsize is None:
        # default font size
        fontsize = 7

    if linewidth is None:
        # default line width
        linewidth = .5

    # font
    plt.rcParams["font.size"] = fontsize

    # axes
    plt.rcParams["axes.linewidth"] = linewidth

    # tick
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # linewidth
    plt.rcParams["lines.linewidth"] = linewidth

    return fig


###
# Make a matrix image arranging brain slices
###
def ndimages(im, params=None):
    imsize = im.shape

    showrgb = 0

    if not params:
        params = {}

    if "axisimage" not in params:
        params["axisimage"] = 0

    if "axisxy" not in params:
        params["axisxy"] = 0

    if len(imsize) == 2:
        bigims = im
    else:  # 5 or 6 dimensions:
        if len(imsize) == 3:
            imnum = imsize[2]
            sx = math.ceil(imnum ** 0.5)
            sy = math.ceil(imnum / sx)
            impadnum = sx * sy - imnum

            im = np.concatenate((im, im[:, :, 0:impadnum] * 0), axis=2)  # 必要性を要調査
            im = np.reshape(im, (imsize[0], imsize[1], sx, sy), order="F")
            imsize = [imsize[0], imsize[1], sx, sy, 1]
        else:
            sx = imsize[2]
            sy = imsize[3]

        bigims = np.ones((sy * imsize[0] + sy, sx * imsize[1] + sx, max(1, showrgb * 3)), dtype=im.dtype)
        bigims = np.squeeze(bigims)
        if len(imsize) == 4:
            imsize = np.append(imsize, 1)

        if len(imsize) == 5:
            imnum = imsize[4]
        else:
            imnum = imsize[4] * imsize[5]
            im = np.reshape(im, (imsize[0], imsize[1], imsize[2], imsize[3], imsize[4] * imsize[5]))

        for _ in range(imnum):
            for ty in range(imsize[3]):
                for tx in range(imsize[2]):
                    x_st = tx * (imsize[1] + 1)
                    x_ed = x_st + (imsize[1])
                    y_st = ty * (imsize[0] + 1)
                    y_ed = y_st + (imsize[0])

                    bigims[y_st:y_ed, x_st:x_ed] = im[:, :, tx, ty]

    return bigims


###
# Save model data to a HDF5 file
###
def save_model_to_hdf5(fname, model_param=None, model_weight=None, model_result=None):
    with h5py.File(fname, "w") as wf:
        if model_param is not None:
            for key in model_param.keys():
                wf.create_dataset("modelParam/" + key, data=model_param[key])

        if model_weight is not None:
            wf.create_dataset("modelWeight", data=model_weight)

        if model_result is not None:
            for key in model_result.keys():
                wf.create_dataset("modelResult/" + key, data=model_result[key])


###
# Get size of clips
###
def get_clip_sizes(seq_ids):
    # Get unique ids of cilp with order preserved
    _, idx = np.unique(seq_ids, return_index=True)
    uniq_ids = seq_ids[np.sort(idx)]

    clip_sizes = []
    for uniq_id in uniq_ids:
        clip_size = seq_ids[seq_ids == uniq_id].shape[0]
        clip_sizes.append(clip_size)

    return clip_sizes


###
# Average weights over delays
###
def average_weight_over_delays(weight, n_delays=None):
    if n_delays is None:
        n_delays = 4

    n_vecs = int(weight.shape[0] / n_delays)
    n_voxs = weight.shape[1]

    mean_weight = np.zeros((n_vecs, n_voxs))

    # Sum weights over delays
    for di in range(n_delays):
        st_id = di * n_vecs
        ed_id = st_id + n_vecs
        mean_weight += weight[st_id:ed_id, :]

    # Average
    mean_weight /= n_delays

    return mean_weight
