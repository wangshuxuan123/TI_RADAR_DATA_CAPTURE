# --coding:utf-8--
import sys

import serial
import numpy as np
import time
from multiprocessing import Process, shared_memory
from multiprocessing.managers import SharedMemoryManager
import matplotlib.pyplot as plt
import pyqtgraph as pg
from threading import Thread
from PyQt5 import QtWidgets
from PyQt5 import QtCore    #要制作关闭按钮，关于动作的类都在QtCore里
from PyQt5.QtGui import QFont   #QtWidgets不包含QFont必须调用QtGui
from PyQt5.QtCore import QCoreApplication
from Radar import ADC_PARAMS

class Vital_GUI():
    def __init__(self, historyLength=400):
        self.historyLength = historyLength
        self.process = Process(target=self.plot, args=())

    def flash_data(self, data):
        i = 0

        while True:
            try:
                breath_data = shared_memory.ShareableList(name='breath_data')
                break
            except:
                continue
        while True:
            if i < self.historyLength:
                data[i] = breath_data[0]
                i += 1
            else:
                data[:-1] = data[1:]
                data[-1] = breath_data[0]
            # time.sleep(0.02)
            time.sleep(0.01)

    def flash_radar_data(self, radar_data):
        sample = np.zeros((ADC_PARAMS['chirps'], ADC_PARAMS['rx'] * ADC_PARAMS['tx'], ADC_PARAMS['samples']),
                          dtype=np.complex)
        while True:
            try:
                existing_shm = shared_memory.SharedMemory(name='frame_buffer')
                frame_buffer_ = np.ndarray(shape=sample.shape, dtype=sample.dtype, buffer=existing_shm.buf)
                break
            except:
                continue

    def close_pyqt(self):
        while True:
            try:
                stop_flag = shared_memory.ShareableList(name='stop_flag')
                break
            except:
                continue
        while True:
            if stop_flag[0] == 1:
                print('stop')

                # self.app.quit()
                self.process.terminate()




    def plot(self):
        # data = array.array('H') # 可动态改变数组的大小,signed short型数组
        data = np.zeros(self.historyLength).__array__('d')  # 把数组长度定下来
        data2 = np.zeros((10,20))
        self.app = pg.mkQApp()
        win = pg.GraphicsWindow(title='采集数据')

        win.resize(1100, 900)
        # plot = pg.plot(title='呼吸波形')
        # plot.plot(data)

        p1 = win.addPlot(left='1',bottom='t', title='呼吸波形')
        p1.showGrid(x=True, y=True)
        p1.setRange(xRange=[0, self.historyLength], yRange=[0, 1], padding=0)
        plot1 = p1.plot(data, pen='b')

        win.nextRow()
        # p2 = win.addPlot(left='2', bottom='t', title='雷达热图')
        # p2.showGrid(x=True, y=True)
        # p2.setRange(xRange=[0, historyLength], yRange=[0, 4500], padding=0)
        # plot2 = p2.plot(data, pen='b')
        # plot2 = p2.imshow(data2)
        th1 = Thread(target=self.flash_data, args=(data, ))
        # th2 = Thread(target=self.close_pyqt, args=())
        th1.start()
        # th2.start()


        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(lambda:plot1.setData(data))
        self.timer.start(10)
        self.app.exec_()


if __name__ == '__main__':
    with SharedMemoryManager() as smm:
        GUI = Vital_GUI()
        GUI.process.start()
        # time.sleep(20)
        # GUI.process.terminate()
        GUI.process.join()

