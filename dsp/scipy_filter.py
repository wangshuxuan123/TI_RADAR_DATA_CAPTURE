from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def highpass_filter(data, N, fs, f, axis=-1):
    """
    Args:
        data:信号
        N: 滤波器阶数
        fs:采样率
        f: 截至频率:
    Returns:
    """
    wn = 2 * f / fs
    b, a = signal.butter(N, wn, 'highpass')
    filterData = signal.filtfilt(b, a, data, axis=axis)
    return filterData


if __name__ == '__main__':
    fs = 1e3
    f1 = 50
    f2 = 100
    ns = 128
    data = np.sin(2 * np.pi * f1 * (1 / fs) * np.arange(ns)) + np.sin(2 * np.pi * f2 * (1 / fs) * np.arange(ns))
    data = data[:, np.newaxis]
    data = np.repeat(data, 64, axis=1)  # [128, 64]
    # fft_data = np.fft.fft(data, axis=0)
    # plt.plot(abs(fft_data[:, 1]))
    # plt.show()
    filter_data = highpass_filter(data, 8, fs, 60.0, axis=0)
    fft_data = np.fft.fft(filter_data, axis=0)
    plt.plot(abs(fft_data[:, 20]))
    plt.show()

    print(1)
