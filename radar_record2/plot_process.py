# --coding:utf-8--
from Radar import ADC_PARAMS
from multiprocessing import shared_memory, Process
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import matplotlib.pyplot as plt
from dsp.utils import Window
# from dataloader.radar_config import RANGE_RESOLUTION, DOPPLER_RESOLUTION, IDLE_TIME, RAMP_END_TIME, MAX_DOPPLER, MAX_RANGE
import dsp
import time
import cv2


def getRawdata(frame_list, flag, flag2, flag3):
    buffer = []
    empty = 0
    i = 0
    sample = np.zeros((ADC_PARAMS['chirps'], ADC_PARAMS['rx'] * ADC_PARAMS['tx'], ADC_PARAMS['samples']),
                      dtype=np.complex)
    raw_frame = sample.copy()
    pre_frame = sample.copy()
    finish = 0  # 判断是否数据读取结束

    while True:
        try:
            existing_shm = shared_memory.SharedMemory(name='frame_buffer')
            frame_buffer_ = np.ndarray(shape=sample.shape, dtype=sample.dtype, buffer=existing_shm.buf)
            break
        except:
            continue
    while True:
        raw_frame[:] = frame_buffer_[:]
        if flag.value:
            # if not ((raw_frame == pre_frame).all()):
                # print((raw_frame == pre_frame).all(), raw_frame[0,0,0], pre_frame[0,0,0])
            i += 1
            flag2.value += 1
            finish = 0
            frame_list.pop(0)
            frame_list.append(raw_frame)
            # print('p2', i, raw_frame[0, 0, 0])
            pre_frame[:] = raw_frame[:]
                # print(i)

def radar_process():
    def Range_Doppler(frame_data, max_range_bin, dim=3):
        radar_cube = dsp.range_processing(frame_data, window_type_1d=Window.HAMMING)
        # print(1, radar_cube.shape)
        # assert radar_cube.shape == (
        #     ADC_PARAMS['chirps'], ADC_PARAMS['tx'] * ADC_PARAMS['rx'], ADC_PARAMS['samples']), "Shape错误"

        # 截取range_bin 0:0.6m
        if dim == 3:
            radar_cube[:, :, max_range_bin:] = 0
        else:
            radar_cube[:, :, :, max_range_bin:] = 0

        # Doppler processing
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=ADC_PARAMS['tx'],
                                                       clutter_removal_enabled=True,
                                                       interleaved=False, window_type_2d=Window.HAMMING,
                                                       high_pass=None)

        det_matrix_vis = np.fft.fftshift(det_matrix, axes=-1)
        return det_matrix_vis

    def Range_Azimuth(frame_data, dim=3):
        radar_cube = dsp.range_processing(frame_data, window_type_1d=Window.HAMMING)

        # Doppler processing
        # det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=ADC_PARAMS['tx'],
        #                                                clutter_removal_enabled=True,
        #                                                interleaved=False, window_type_2d=Window.HAMMING,
        #                                                high_pass=None)
        # print(aoa_input.shape)
        # azimuth processing
        _, range_azimuth_vis = dsp.angle_processing(radar_cube.transpose(2,1,0), fft_size=ADC_PARAMS['angles'], static_remove=False, radar_type='1642')
        # print(range_azimuth_vis.shape)

        range_azimuth_vis = np.fft.fftshift(range_azimuth_vis, axes=-1)
        return range_azimuth_vis

    # range_doppler

    # # 参数
    max_range_bin = 256
    # 共享内存
    sample = np.zeros((ADC_PARAMS['samples'], ADC_PARAMS['angles']),
                      dtype=np.float)
    shm = shared_memory.SharedMemory(name='vis_buffer', create=True, size=sample.nbytes)
    vis_buffer = np.ndarray(sample.shape, dtype=sample.dtype, buffer=shm.buf)

    sample2 = np.zeros((ADC_PARAMS['chirps'], ADC_PARAMS['rx'] * ADC_PARAMS['tx'], ADC_PARAMS['samples']),
                       dtype=np.complex)
    raw_frame = sample2.copy()
    while True:
        try:
            existing_shm = shared_memory.SharedMemory(name='frame_buffer')
            frame_buffer_ = np.ndarray(shape=sample2.shape, dtype=sample2.dtype, buffer=existing_shm.buf)
            plot_flag = shared_memory.ShareableList(name='plot_flag')

            break
        except:
            continue

    while True:
        raw_frame[:] = frame_buffer_[:]
        # if not ((raw_frame == sample_frames).all()):
        vis_buffer[:] = Range_Azimuth(raw_frame, dim=3)
        if plot_flag[0] == 2:
            break



def plot_fig():
    fig = plt.figure()
    # max_range_bin = int(0.6 // RANGE_RESOLUTION) + 1
    sample = np.zeros((ADC_PARAMS['samples'], ADC_PARAMS['angles']),
                      dtype=np.float)
    vis_frame = sample.copy()


    # x_ticks = np.around(RANGE_RESOLUTION * (np.arange(ADC_PARAMS['chirps']) - ADC_PARAMS['chirps']//2+1), 2)


    # x_ticks2 = np.around(np.arange(-MAX_DOPPLER, MAX_DOPPLER, DOPPLER_RESOLUTION*8), 2)
    # x_ticks1 = np.arange(0, ADC_PARAMS['chirps'], ADC_PARAMS['chirps']//len(x_ticks2))
    # y_ticks2 = np.around(np.arange(0, MAX_RANGE, RANGE_RESOLUTION*16), 2)
    # y_ticks1 = np.arange(0, 256,16)
    # print(x_ticks)
    while True:
        try:
            existing_shm = shared_memory.SharedMemory(name='vis_buffer')
            vis_buffer_ = np.ndarray(shape=sample.shape, dtype=sample.dtype, buffer=existing_shm.buf)
            break
        except:
            continue
    while True:
        try:
            plot_flag = shared_memory.ShareableList(name='plot_flag')
        except:
            continue
        if plot_flag[0] == 1:
            vis_frame[:] = vis_buffer_
            # vis_frame_ = transpose(vis_frame[:max_range_bin, :])
            plt.imshow(vis_frame)
            # plt.title("Range-Doppler plot" + str(frames))

            # plt.xticks(x_ticks1,x_ticks2, fontsize=5, rotation=70)
            # plt.yticks(y_ticks1, y_ticks2, fontsize=5)
            plt.ylabel('range: m',fontsize=5)
            plt.xlabel('doppler: m/s',fontsize=5)
            plt.draw()
            plt.pause(0.001)
            plt.clf()

        elif plot_flag[0] == 2:
            break

def normalize(map, cv_w=512, cv_h=1024, max_v=40):

    np.where(map >= max_v, max_v, map)
    map = np.rint((map / max_v) * 255)
    # map = map[:, :, np.newaxis]
    map = np.array(map,dtype=np.uint8)
    map = cv2.resize(map, (cv_w, cv_h), interpolation=cv2.INTER_CUBIC)
    map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
    map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
    # print(map)
    return map

def cv_plot(cv_w=512, cv_h=1024, max_v=40):
    sample = np.zeros((ADC_PARAMS['samples'], ADC_PARAMS['angles']),
                      dtype=np.float)
    vis_frame = sample.copy()
    while True:
        try:
            existing_shm = shared_memory.SharedMemory(name='vis_buffer')
            vis_buffer_ = np.ndarray(shape=sample.shape, dtype=sample.dtype, buffer=existing_shm.buf)
            break
        except:
            continue
    while True:
        try:
            plot_flag = shared_memory.ShareableList(name='plot_flag')
            break
        except:
            continue
    while True:
        if plot_flag[0] == 1:
            vis_frame[:] = vis_buffer_[:]
            # print(vis_frame)
            vis = normalize(vis_frame, cv_w, cv_h, max_v)
            cv2.imshow('Range_Angle', vis)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            time.sleep(0.01)
        if plot_flag[0] == 2:
            break

if __name__ == '__main__':
    with SharedMemoryManager() as shm:
        process = Process(target=radar_process, args=())
        process2 = Process(target=plot_fig, args=())
        process.start()
        process2.start()
        process.join()
        process2.join()
