# --coding:utf-8--
# --coding:utf-8--
import numpy as np
import socket
import struct
import codecs
import time
from enum import Enum
from multiprocessing import Process, Manager, Lock, Array, Queue, shared_memory, Value
from multiprocessing.managers import SharedMemoryManager
import ctypes
import matplotlib.pyplot as plt
from dsp.utils import Window
import dsp
# from dataloader.radar_config import RANGE_RESOLUTION, DOPPLER_RESOLUTION, IDLE_TIME, RAMP_END_TIME, MAX_DOPPLER, MAX_RANGE
from collections import Counter
import serial
import threading


ADC_PARAMS = {'chirps': 16,
              'rx': 4,
              'tx': 2,
              'IQ': 2,
              'angles': 16,
              'samples': 256,
              'bytes': 2,
              'frames': 200}
# fs = 1 / ((RAMP_END_TIME + IDLE_TIME) * 1e-6)
# f = fs // ADC_PARAMS['chirps'] * 2
# STATIC
MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456

BYTES_IN_FRAME = (ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['tx'] *
                  ADC_PARAMS['samples'] * ADC_PARAMS['IQ'] * ADC_PARAMS['bytes'])
BYTES_IN_TOTAL_FRAME = BYTES_IN_FRAME * ADC_PARAMS['frames']
INT16_IN_FRAME = BYTES_IN_FRAME // 2
INT16_IN_PACKET = BYTES_IN_PACKET // 2
BYTES_IN_FRAME_CLIPPED = (BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
PACKETS_IN_FRAME = BYTES_IN_FRAME / BYTES_IN_PACKET
PACKETS_IN_FRAME_CLIPPED = BYTES_IN_FRAME // BYTES_IN_PACKET

BYTES_IN_CHIRP = (ADC_PARAMS['rx'] * ADC_PARAMS['tx'] * ADC_PARAMS['samples'] * ADC_PARAMS['IQ'] * ADC_PARAMS['bytes'])

TOTAL_PACKET_NUM = BYTES_IN_TOTAL_FRAME // BYTES_IN_PACKET + 1


class DCA1000():
    def __init__(self, start_flag, stop_flag, save_dir, static_ip='192.168.33.31', adc_ip='192.168.33.180', cmd_port=4096,
                 adc_port=4098, serial_port='COM7', cfg_file='cfg/profile_1642.cfg'):
        # threading.Thread.__init__(self)
        # 数据存放路径
        self.save_dir = save_dir
        self.start_flag = start_flag
        self.stop_flag = stop_flag
        # 设置ip和端口
        self.cfg_dest = (adc_ip, cmd_port)
        self.cfg_recv = (static_ip, cmd_port)
        self.adc_recv = (static_ip, adc_port)

        # 设置UDP套接字
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        # 设置串口
        self.serial_port = serial_port
        self.cfg_file = cfg_file
        # 设置接收缓冲区大小
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 ** 30)
        # recv_buff = self.data_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        # print(recv_buff)
        self.config_socket.bind(self.cfg_recv)
        self.data_socket.bind(self.adc_recv)
        self.buffer = b''  # 将每个数据包累加直到满一帧

        # 存放处理好的frame数据
        self.frame_buffer = []
        # self.queue = manager.Queue(maxsize=2**20)
        # self.buffer2 = Array(ctypes.c_char, 2 ** 20)
        # self.frames_buffer = [0 for _ in range(ADC_PARAMS['frames'])]  # 缓冲区存放帧数据
        self.frames_buffer = []
        self.lost_packets = None
        self.count = 1
        self.pre_packet_num = 0
        self.first = True  # 第一个循环判断第一个包是否读取正确，若正确置为False
        self.finish = False  # 判断是否读取到了最后一个包

        self.data_socket.settimeout(10)

        self.process = Process(target=self.run, args=(10, ))

    def _read_data_packet(self):
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.int16)
        return packet_num, byte_count, packet_data.tolist()

    def send_radar_config(self):
        # 通过串口配置雷达参数
        CLIport = serial.Serial(self.serial_port, 115200, timeout=1)
        if CLIport.is_open:
            config = [line.strip('\r\n') for line in open(self.cfg_file)]
            for i in config:
                # Skip empty line
                if (i == ''):
                    continue
                # Skip comment line
                if (i[0] == '%'):
                    continue
                # Stop on sensorStart command
                # if ('calibData' in i):
                #     # print(11111111111)
                #     self.flag.value = 1
                CLIport.write((i + '\n').encode())
                # print('>>> ' + i)
                time.sleep(0.01)
        CLIport.close()


    def run(self, timeout=25):
        self.data_socket.settimeout(timeout)
        start_time = time.time()
        i = 0
        # total_loss = 0
        buffer = []
        zero_buffer = [0] * INT16_IN_PACKET
        # sample = np.zeros((ADC_PARAMS['chirps'], ADC_PARAMS['rx'] * ADC_PARAMS['tx'], ADC_PARAMS['samples']),
        #                   dtype=np.complex)
        # shm = shared_memory.SharedMemory(name='frame_buffer', create=True, size=sample.nbytes)
        # frame_buffer = np.ndarray(sample.shape, dtype=sample.dtype, buffer=shm.buf)
        # frame_buffer = []
        frame_num = 0
        # 共享内存
        sample = np.zeros((ADC_PARAMS['chirps'], ADC_PARAMS['rx'] * ADC_PARAMS['tx'], ADC_PARAMS['samples']),
                          dtype=np.complex)
        shm = shared_memory.SharedMemory(name='frame_buffer', create=True, size=sample.nbytes)
        share_frame_buffer = np.ndarray(sample.shape, dtype=sample.dtype, buffer=shm.buf)

        # 开始画图的标志
        plot_flag = shared_memory.ShareableList([0], name='plot_flag')


        print('start======================')
        self.start_flag.value = 1 # 启动其他的设备
        print('雷达设备启动')
        self.send_radar_config()

        while True:
            try:
                # 判断缓冲区字节数是否满足一帧的大小，若满足则返回一帧数据
                # if len(self.buffer) >= BYTES_IN_FRAME:
                #     # ret_frame = np.zeros(INT16_IN_FRAME, dtype=np.int16)
                #     ret_frame = np.frombuffer(self.buffer[:BYTES_IN_FRAME], dtype=np.int16)
                #     self.buffer = self.buffer[BYTES_IN_FRAME:]
                packet_num, byte_count, packet_raw_data = self._read_data_packet()
                # print(packet_num)
                if i == 0:
                    # 第一个包
                    buffer += packet_raw_data
                else:
                    # total_loss += packet_num - pre_packet_num - 1
                    buffer += (packet_num - pre_packet_num - 1) * zero_buffer + packet_raw_data
                    # if packet_num != pre_packet + 1:
                    #     print(packet_num - pre_packet - 1)
                pre_packet_num = packet_num
                i += 1
                # print(len(buffer))
                if len(buffer) >= INT16_IN_FRAME:
                    # print(buffer[:INT16_IN_FRAME])
                    if Counter(buffer[:INT16_IN_FRAME])[0] >= INT16_IN_FRAME // 2:
                        # 如果某一帧丢包超过一半，则copy前一帧数据
                        frame_num += 1
                        self.frame_buffer.append(self.frame_buffer[-1])
                        buffer = buffer[INT16_IN_FRAME:]
                    else:
                        frame_num += 1
                        frame = self.organize(np.array(buffer[:INT16_IN_FRAME]), ADC_PARAMS['chirps'], ADC_PARAMS['tx'],
                                              ADC_PARAMS['rx'], ADC_PARAMS['samples'])
                        self.frame_buffer.append(frame)

                        buffer = buffer[INT16_IN_FRAME:]

                        # print('p1', frame_num, frame[0, 0, 0])
                # print(len(packet_raw_data))
                    share_frame_buffer[:] = frame / 2**15
                    plot_flag[0] = 1
                if len(packet_raw_data) != INT16_IN_PACKET:
                    print('雷达耗时', time.time() - start_time)
                    self.stop_flag.value = 1  # 读取完成标志
                    print('数据接受完成')
                    break
            except:
                self.data_socket.close()
                break
                # print(i)

        # 处理多余的帧数据
        if len(buffer) % INT16_IN_FRAME != 0:
            raise Exception('多余的帧数据不能被整除')
        while len(buffer) > 0:
            # print("还有",len(buffer))
            frame = buffer[:INT16_IN_FRAME]
            self.frame_buffer.append(frame)
            buffer = buffer[INT16_IN_FRAME:]
            frame_num += 1

        # flag3.value = 1  # 读取完成标志
        try:
            print('最后一个包INT16长度%d' % len(packet_raw_data))
            print("数据获取完成，总共获得%d个包\n" % i)
            # print('丢了%d个包' % total_loss)
            print('丢了%d个包' % (TOTAL_PACKET_NUM - i))
            # print(packet_num)

            print('总帧数%d' % frame_num)
        except:
            raise Exception('数据录取有误')
        # print('雷达耗时:', time.time()-total_start_time)
        # 结束画图
        plot_flag[0] = 2

        np.save(self.save_dir, np.array(self.frame_buffer))
        print(np.array(self.frame_buffer).shape)

    def get_frame_buffer(self):
        return np.array(self.frame_buffer)
    def organize(self, raw_frame, num_chirps, num_tx, num_rx, num_samples):
        """
        :param raw_frame:  [INT16_IN_FRAME]  INT16_IN_FRAME = num_chirps * num_rx * num_samples * IQ
        :param num_chirps: 一帧数据包含的chirps 即 num_chirps = loops * tx
        :param num_rx:
        :param num_samples:
        :return: [num_chirps, num_rx, num_samples]
        """
        ret = np.zeros(len(raw_frame) // 2, dtype=complex)
        # ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
        # ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
        ret[0::2] = 1j * raw_frame[0::4] + raw_frame[2::4]
        ret[1::2] = 1j * raw_frame[1::4] + raw_frame[3::4]
        return ret.reshape((num_chirps, num_tx * num_rx, num_samples))


if __name__ == '__main__':
    flag = Value('d', 0)
    radar = DCA1000(flag, save_dir='a.npy')
    radar.process.start()
    radar.process.join()

    # data = radar.get_frame_buffer()
    # print(data.shape)


