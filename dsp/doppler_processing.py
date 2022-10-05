import numpy as np
from . import compensation
from . import utils
from .scipy_filter import highpass_filter

def doppler_resolution(band_width, start_freq_const=77, ramp_end_time=62, idle_time_const=100, num_loops_per_frame=128,
                       num_tx_antennas=3):
    """Calculate the doppler resolution for the given radar configuration.
    Args:
        start_freq_const (int): Frequency chirp starting point.
        ramp_end_time (int): Frequency chirp end point.
        idle_time_const (int): Idle time between chirps.
        band_width (float): Radar config bandwidth.
        num_loops_per_frame (int): The number of loops in each frame.
        num_tx_antennas (int): The number of transmitting antennas (tx) on the radar.
    Returns:
        doppler_resolution (float): The doppler resolution for the given radar configuration.
    """

    light_speed_meter_per_sec = 299792458

    center_frequency = start_freq_const * 1e9 + band_width / 2
    chirp_interval = (ramp_end_time + idle_time_const) * 1e-6
    doppler_resolution = light_speed_meter_per_sec / (
            2 * num_loops_per_frame * num_tx_antennas * center_frequency * chirp_interval)

    return doppler_resolution


def separate_tx(signal, num_tx, vx_axis=1, axis=0):
    """Separate interleaved radar data from separate TX along a certain axis to account for TDM radars.
    Args:
        signal (ndarray): Received signal.
        num_tx (int): Number of transmit antennas.
        vx_axis (int): Axis in which to accumulate the separated data.
        axis (int): Axis in which the data is interleaved.
    Returns:
        ndarray: Separated received data in the
    """
    # Reorder the axes
    reordering = np.arange(len(signal.shape))
    if len(reordering) == 3:
        reordering[0] = axis
        reordering[axis] = 0
        signal = signal.transpose(reordering)
        out = np.concatenate([signal[i::num_tx, ...] for i in range(num_tx)], axis=vx_axis)
    elif len(reordering) == 4:
        reordering[0] = axis+1
        reordering[axis+1] = 0
        signal = signal.transpose(reordering)
        out = np.concatenate([signal[i::num_tx, ...] for i in range(num_tx)], axis=vx_axis+1)
    return out.transpose(reordering)


def doppler_processing(radar_cube,
                       num_tx_antennas=3,
                       high_pass=None,
                       clutter_removal_enabled=True,
                       interleaved=True,
                       window_type_2d=None,
                       accumulate=True):
    """Perform 2D FFT on the radar_cube.
    Interleave the radar_cube, perform optional windowing and 2D FFT on the radar_cube. Optional antenna couping
    signature removal can also be performed right before 2D FFT. In constrast to the original TI codes, CFAR and peak
    grouping are intentionally separated with 2D FFT for the easiness of debugging.
    Args:
        radar_cube (ndarray): Output of the 1D FFT. If not interleaved beforehand, it has the shape of
            (numChirpsPerFrame, numRxAntennas, numRangeBins). Otherwise, it has the shape of
            (numRangeBins, numVirtualAntennas, num_doppler_bins). It is assumed that after interleaving the doppler
            dimension is located at the last axis.
        num_tx_antennas (int): Number of transmitter antennas. This affects how interleaving is performed.
        clutter_removal_enabled (boolean): Flag to enable naive clutter removal.
        interleaved (boolean): If the input radar_cube is interleaved before passing in. The default radar_cube is not
            interleaved, i.e. has the shape of (numChirpsPerFrame, numRxAntennas, numRangeBins). The interleaving
            process will transform it such that it becomes (numRangeBins, numVirtualAntennas, num_doppler_bins). Note
            that this interleaving is only applicable to TDM radar, i.e. each tx emits the chirp sequentially.
        window_type_2d (mmwave.dsp.utils.Window): Optional windowing type before doppler FFT.
        accumulate (boolean): Flag to reduce the numVirtualAntennas dimension.

    Returns:
        detMatrix (ndarray): (numRangeBins, num_doppler_bins) complete range-dopper information. Original datatype is
                             uint16_t. Note that azimuthStaticHeatMap can be extracted from zero-doppler index for
                             visualization.
        aoa_input (ndarray): (numRangeBins, numVirtualAntennas, num_doppler_bins) ADC data reorganized by vrx instead of
                             physical rx.
    """

    if interleaved:
        # radar_cube is interleaved in the first dimension (for 2 tx and 0-based indexing, odd are the chirps from tx1,
        # and even are from tx2) so it becomes ( , numVirtualAntennas, numADCSamples), where
        # numChirpsPerFrame = num_doppler_bins * num_tx_antennas as designed.
        # Antennas associated to tx1 (Ping) are 0:4 and to tx2 (Pong) are 5:8.
        fft2d_in = separate_tx(radar_cube, num_tx_antennas, vx_axis=1, axis=0)
        origin_fft2d_in = fft2d_in
    else:
        fft2d_in = radar_cube



    # (Optional) Static Clutter Removal
    if clutter_removal_enabled:
        fft2d_in = compensation.clutter_removal(fft2d_in, axis=0)
    # 高通滤波，滤除速度较小的部分
    if high_pass and len(fft2d_in.shape) == 3:
        fft2d_in = highpass_filter(fft2d_in,high_pass['N'], high_pass['fs'], high_pass['f'], axis=0)
    elif high_pass and len(fft2d_in.shape) == 4:
        fft2d_in = highpass_filter(fft2d_in, high_pass['N'], high_pass['fs'], high_pass['f'], axis=1)
    # transpose to (numRangeBins, numVirtualAntennas, num_doppler_bins)
    if len(fft2d_in.shape) == 3:
        fft2d_in = np.transpose(fft2d_in, axes=(2, 1, 0))
    elif len(fft2d_in.shape) == 4:
        fft2d_in = np.transpose(fft2d_in, axes=(0, 3, 2, 1))

    # Windowing 16x32
    if window_type_2d:
        fft2d_in = utils.windowing(fft2d_in, window_type_2d, axis=-1)

    # It is assumed that doppler is at the last axis.
    # FFT 32x32
    fft2d_out = np.fft.fft(fft2d_in)
    aoa_input = fft2d_out

    # Save zero-Doppler as azimuthStaticHeatMap, watch out for the bit shift in
    # original code.

    # Log_2 Absolute Value
    fft2d_abs = np.abs(fft2d_out)

    # Accumulate
    if accumulate:
        if len(fft2d_in.shape) == 3:
            return np.mean(fft2d_abs, axis=1), aoa_input
            #return fft2d_log_abs[:,1,:], aoa_input
        elif len(fft2d_in.shape) == 4:
            return np.mean(fft2d_abs, axis=2), aoa_input
    else:
        return fft2d_abs, aoa_input


def doppler_estimation(radar_cube,
                       beam_weights,
                       num_tx_antennas=2,
                       clutter_removal_enabled=False,
                       interleaved=False,
                       window_type_2d=None):
    """Perform doppler estimation on the weighted sum of range FFT output across all virtual antennas.

    In contrast to directly computing doppler FFT from the output of range FFT, this function combines it across all
    the virtual receivers first using the weights generated from beamforming. Then FFT is performed and argmax is taken
    across each doppler axis to return the indices of max doppler values.

    Args:
        radar_cube (ndarray): Output of the 1D FFT with only ranges on detected objects. If not interleaved beforehand,
            it has the shape of (numChirpsPerFrame, numRxAntennas, numDetObjs). Otherwise, it has the shape of
            (numDetObjs, numVirtualAntennas, num_doppler_bins). It is assumed that after interleaving the doppler
            dimension is located at the last axis.
        beam_weights (ndarray): Weights to sum up the radar_cube across the virtual receivers. It is from the
                                beam-forming and has the shape of (numVirtualAntennas, numDetObjs)
        num_tx_antennas (int): Number of transmitter antennas. This affects how interleaving is performed.
        clutter_removal_enabled (boolean): Flag to enable naive clutter removal.
        interleaved (boolean): If the input radar_cube is interleaved before passing in. The default radar_cube is not
            interleaved, i.e. has the shape of (numChirpsPerFrame, numRxAntennas, numDetObjs). The interleaveing process
            will transform it such that it becomes (numDetObjs, numVirtualAntennas, num_doppler_bins). Note that this
            interleaving is only appliable to TDM radar, i.e. each tx emits the chirp sequentially.
        window_type_2d (string): Optional windowing type before doppler FFT.

    Returns:
        doppler_est (ndarray): (numDetObjs) Doppler index for each detected objects. Positive index means moving away
                               from radar while negative index means moving towards the radar.
    """
    fft2d_in = None
    if not interleaved:
        num_doppler_bins = radar_cube.shape[0] / num_tx_antennas
        # radar_cube is interleaved in the first dimension (for 2 tx and 0-based indexing, odd are the chirps from tx1,
        # and even are from tx2) so it becomes (num_doppler_bins, numVirtualAntennas, numADCSamples), where
        # numChirpsPerFrame = num_doppler_bins * num_tx_antennas as designed.
        # Antennas associated to tx1 (Ping) are 0:4 and to tx2 (Pong) are 5:8.
        if num_tx_antennas == 2:
            fft2d_in = np.concatenate((radar_cube[0::2, ...], radar_cube[1::2, ...]), axis=1)
        elif num_tx_antennas == 3:
            fft2d_in = np.concatenate((radar_cube[0::3, ...], radar_cube[1::3, ...], radar_cube[2::3, ...]), axis=1)

        # transpose to (numRangeBins, numVirtualAntennas, num_doppler_bins)
        fft2d_in = np.transpose(fft2d_in, axes=(2, 1, 0))
    else:
        num_doppler_bins = radar_cube.shape[2]

    # (Optional) Static Clutter Removal
    if clutter_removal_enabled:
        fft2d_in = compensation.clutter_removal(fft2d_in)

    # Weighted sum across all virtual receivers.
    fft2d_in = np.einsum('ijk,jk->ik', fft2d_in, beam_weights)

    # Windowing 16x32
    if window_type_2d:
        fft2d_in = utils.windowing(fft2d_in, window_type_2d, axis=1)

    # It is assumed that doppler is at the last axis.
    # FFT 32x32
    doppler_est = np.fft.fft(fft2d_in)
    doppler_est = np.argmax(doppler_est, axis=1)
    doppler_est[doppler_est[:] >= num_doppler_bins] -= num_doppler_bins * 2

    return doppler_est