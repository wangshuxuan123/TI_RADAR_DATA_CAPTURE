# --coding:utf-8--

import serial
import numpy as np
import time
from multiprocessing import Process, shared_memory
from multiprocessing.managers import SharedMemoryManager
import matplotlib.pyplot as plt
import pyqtgraph as pg
from threading import Thread

start_command = 'FF C8 03 A3 A0'

class Heart_Record():
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

        while True:
            if self.start_flag.value:
                print('心率设备启动')
                self.send_start_command()
                break
            else:
                time.sleep(0.01)
                continue
        start_time = time.time()
        while self.stop_flag.value != 1:
            receive_data = self.serial.read().hex()
            current_time = time.time() - start_time
            frame_data = []
            if receive_data == 'ff': # 帧起始
                for i in range(6):
                    frame_data.append(self.serial.read())
                MB = frame_data[-2] + frame_data[-1]
                heart_rate = int(MB.hex(), 16)
                self.frames.append([current_time, heart_rate])
                # print(current_time, heart_rate)

        print('心跳设备耗时: ', time.time()-start_time)
        self.serial.close()
        np.save(self.save_dir, np.array(self.frames))
        # plt.plot(self.frames[1:])
        # plt.show()

if __name__ == '__main__':
    # with SharedMemoryManager() as smm:
    #     heart_record = Heart_Record()
    #     process = Process(target=heart_record.run, args=())
    #     process.start()
    #     process.join()
    heart_record = Heart_Record()
    heart_record.run()