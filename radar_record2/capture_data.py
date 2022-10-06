# --coding:utf-8--
import time

import numpy as np

from Breath import Breath_Record
from Radar import DCA1000, ADC_PARAMS
from Heart import Heart_Record
from multiprocessing import Value, shared_memory, Process
from multiprocessing.managers import SharedMemoryManager
import os
from plot_process import plot_fig, radar_process, cv_plot

# config
sample_time = 10 # 采集时间 s


# radar config
radar_config_port = 'COM7'
radar_frame_rate = 20 # 雷达帧率率 1s/50ms 一帧50ms
radar_frame = sample_time * radar_frame_rate  # 雷达采样帧数

# breath config
breath_port = 'COM30'
breath_fs = 50 # 采样率 一帧采样50个点
breath_samples = sample_time * breath_fs  # 采样总数

# heart config
heart_port = 'COM25'
heart_sample_time = sample_time # 心跳采集时间

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

if __name__ == '__main__':
    save_dir = '../data'
    person = 'wsx2'
    counts = 0
    save_dir = os.path.join(save_dir, person)  # './wsx/'
    mkdir(save_dir)
    save_dir = os.path.join(save_dir, str(counts).zfill(3)) # './wsx/000'
    mkdir(save_dir)

    with SharedMemoryManager() as smm:
        # 标志位
        start_flag = Value('d', 0)  # 判断雷达是否开始采集数据
        stop_flag = Value('d', 0) # 雷达采集结束标志
        #共享内存
        breath_data = shared_memory.ShareableList([0], name='breath_data')
        # sample = np.zeros((ADC_PARAMS['chirps'], ADC_PARAMS['rx'] * ADC_PARAMS['tx'], ADC_PARAMS['samples']),
        #                   dtype=np.complex)
        # shm = shared_memory.SharedMemory(name='radar_data', create=True, size=sample.nbytes)


        radar = DCA1000(start_flag=start_flag, stop_flag=stop_flag,serial_port=radar_config_port, save_dir=os.path.join(save_dir, 'radar.npy'))
        breath = Breath_Record(port=breath_port, start_flag=start_flag, stop_flag=stop_flag, save_dir=os.path.join(save_dir, 'breath.npy'))
        heart = Heart_Record(port=heart_port, start_flag=start_flag, stop_flag=stop_flag, save_dir=os.path.join(save_dir, 'heart.npy'))

        radar_process = Process(target=radar_process, args=())
        plot_process = Process(target=cv_plot, args=(512, 1024, 40))
        radar.process.start()
        # breath.process.start()
        # heart.process.start()
        radar_process.start()
        plot_process.start()
        # time.sleep(1)
        # flag.value = 1
        radar.process.join()
        # breath.process.join()
        # heart.process.join()
        radar_process.join()
        plot_process.join()
