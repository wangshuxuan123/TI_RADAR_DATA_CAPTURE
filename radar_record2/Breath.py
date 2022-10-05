# --coding:utf-8--
import array

import serial
import numpy as np
import time
from multiprocessing import Process, shared_memory
from multiprocessing.managers import SharedMemoryManager
import matplotlib.pyplot as plt
import pyqtgraph as pg
from threading import Thread

start_command = 'FF CC 03 A3 A0'
# x = [bytes.fromhex('0001').hex()]
# print(1)
# xx = int((b'\x01' + b'\x01').hex(),16)
#
# print(1)
class Breath_Record():
    def __init__(self, start_flag=None, stop_flag=None, save_dir=None, port='COM25', amp=10):
        self.port = port
        self.amp = amp
        self.start_flag = start_flag
        self.stop_flag = stop_flag
        self.save_dir = save_dir
        self.frames = []
        self.process = Process(target=self.run, args=())
    def amplitude_adjust(self, amp):
        amp =hex(amp) # 幅值转16进制
        if len(amp) == 3:
            amp = '0' + amp[-1]
        else:
            amp = amp[-(len(amp)-2):]

        command =['04', 'A4', amp] # 长度、命令、参数16进制
        CKSUM = hex(sum(int(i, 16) for i in command))[-2:] # 求和取低8字节
        send_command = '04 A4 04' + ' ' + CKSUM + ' ' + amp
        while True:
            if self.serial.is_open:
                self.serial.write(bytes.fromhex(send_command))
                break
            else:
                continue

    def send_start_command(self):
        self.serial.write(bytes.fromhex(start_command))


    def run(self):
        self.serial = serial.Serial(port=self.port, baudrate=115200, timeout=None)
        self.serial.flushInput() # 清空缓冲区

        # self.amplitude_adjust(amp=self.amp)
        # 共享内存
        breath_data = shared_memory.ShareableList(name='breath_data')

        # 发送配置命令
        while True:
            if self.start_flag.value:
                print('呼吸设备启动')
                self.send_start_command()
                break
            else:
                time.sleep(0.01)
                continue

        start_time = time.time()
        while self.stop_flag.value != 1:
            receive_data = self.serial.read().hex()
            frame_data = []
            if receive_data == 'ff': # 帧起始
                for i in range(6):
                    frame_data.append(self.serial.read())
                MB = frame_data[-2] + frame_data[-1]
                amp = int(MB.hex(), 16)
                self.frames.append(amp)
                breath_data[0] = amp/1000
        print('呼吸设备耗时:', time.time() - start_time)
        np.save(self.save_dir, np.array(self.frames))
        self.serial.close()

        # plt.plot(self.frames[1:])
        # plt.show()


# def flash_data(breath_data):
#     i=0
#     while True:
#         breath_data[0] = i
#         i+=1
#         time.sleep(0.01)




class plot_breath():
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

    def plot(self):
        # data = array.array('H') # 可动态改变数组的大小,signed short型数组
        data = np.zeros(self.historyLength).__array__('d')  # 把数组长度定下来

        app = pg.mkQApp()
        win = pg.GraphicsWindow(title='采集数据')

        win.resize(1100, 900)
        # plot = pg.plot(title='呼吸波形')
        # plot.plot(data)
        p1 = win.addPlot(left='1',bottom='t', title='呼吸波形')
        p1.showGrid(x=True, y=True)
        p1.setRange(xRange=[0, self.historyLength], yRange=[0, 1], padding=0)
        plot1 = p1.plot(data, pen='b')
        # win.nextRow()
        # p2 = win.addPlot(left='2', bottom='t', title='采集节点2波形')
        # p2.showGrid(x=True, y=True)
        # p2.setRange(xRange=[0, historyLength], yRange=[0, 4500], padding=0)
        # plot2 = p2.plot(data, pen='b')
        th1 = Thread(target=self.flash_data, args=(data, ))
        th1.start()
        timer = pg.QtCore.QTimer()
        timer.timeout.connect(lambda:plot1.setData(data))
        timer.start(10)
        app.exec_()



if __name__ == '__main__':
    with SharedMemoryManager() as smm:
        start_flag = shared_memory.ShareableList([0], name='start_flag')
        breath_data = shared_memory.ShareableList([0], name='breath_data')
        # stop_flag = shared_memory.ShareableList([0], name='stop_flag')
        breath_record = Breath_Record()
        process = Process(target=breath_record.run, args=())
        # plot_b = plot_breath()
        # process2 = plot_b.process
        process.start()
        # process2.start()
        time.sleep(2)
        start_flag[0] = 1
        process.join()


        # process2.join()
    #     process = Process(target=flash_data, args=(breath_data, ))
    #     plot_b = plot_breath()
        # process2 = plot_b.process
        # process.start()
        # process2.start()
        # process.join()
        # process2.join()





