# --coding:utf-8--
import numpy as np
import math
from numpy import linalg as la
def get_max_range_bin(range_profile, axis=0):
    range_profile_mean = np.mean(np.abs(range_profile), axis=axis)
    max_range_bin = np.argmax(range_profile_mean)
    return max_range_bin

def my_unwrap(phase_data):
    for i in range(1, phase_data.shape[-1]):
        if phase_data[i] - phase_data[i-1] > np.pi:
            phase_data[i] -= 2*np.pi
        elif phase_data[i] - phase_data[i-1] < -np.pi:
            phase_data[i] += 2*np.pi
        else:
            continue
    return phase_data

def dc_remove(range_profile):
    """
    Args:
        range_profile: [slow time, fast time]
    Returns:
    """
    range_profile[:, 0] -= np.average(range_profile[:, 0])
    range_profile[:, -1] -= np.average(range_profile[:, -1])
    return range_profile

def phase_extract(imag_datas, real_datas):
    phase_datas = np.zeros_like(imag_datas)
    for i in range(imag_datas.shape[0]):
        for j in range(imag_datas.shape[1]):
            phase_datas[i][j] = math.atan2(imag_datas[i][j], real_datas[i][j])
    # phase_datas = np.unwrap(phase_datas, axis=-1)
    for i in range(phase_datas.shape[0]):
        phase_datas[i, :] = np.unwrap(phase_datas[i, :])

    return phase_datas

def background_sub(range_profile, lmd=0.9):
    """
    Args:
        range_profile: [slow time , fast time]
    Returns:
    """
    range_profile_sub = np.zeros_like(range_profile)
    range_profile_sub[0, :] = range_profile[0, :]
    Bm = range_profile[0, :] # 初始背景
    for i in range(1, range_profile.shape[0]):
        range_profile_sub[i, :] = range_profile[i, :] - Bm
        Bm = lmd * Bm + (1 - lmd) * range_profile[i, :]
    return range_profile_sub

def svd(range_profile):
    # if range_profile.shape[0] < range_profile.shape[1]:
    #     range_profile = range_profile.T
    # mask = np.zeros()
    # mask[0, 0] =
    m, n = range_profile.shape[0], range_profile.shape[1]
    u, s, vh = la.svd(range_profile)
    new_s = np.zeros((m, n))
    for i in range(min(m, n)):
        new_s[i, i] = s[i]
    # s = s & mask
    new_range_profile = np.dot(np.dot(u, new_s), vh)
    # if range_profile.shape[0] < range_profile.shape[1]:
    #     new_range_profile = new_range_profile.T
    return new_range_profile

