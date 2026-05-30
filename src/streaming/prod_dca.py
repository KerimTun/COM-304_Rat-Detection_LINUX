import numpy as np
import queue
import signal

import time
from datetime import datetime
import os

from mmwave.dataloader.adc import DCA1000
from processing.processing import process_frame, process_frame_2d, beamform_2d, beamform_2d_s
from utils.utils import get_ant_pos_2d
from processing.processing import compute_dbscan




def producer_real_time_1843_SAVE_DOPPLER(q, cfg_radar, cfg_cfar, config_port, data_port, static_ip, system_ip, stop_event, exp_name, exp_path):
    """
    Realtime producer for one radar/DCA1000 stream.

    This function keeps project_3's dual-radar networking and raw recording,
    but changes the per-radar signal chain to the realtime rat-detection path:

        ADC frame -> range FFT -> optional previous-frame subtraction ->
        optional Doppler moving-target notch -> dense beamforming ->
        frame-wise normalization -> optional post-beamforming CFAR.

    The consumer receives normalized dense beamformed polar maps as:
        ("bev", bf_output)
    """

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    r_idxs          =   np.asarray(cfg_radar["range_idx"], dtype=int)
    num_tx          =   int(cfg_radar["num_tx"])
    num_rx          =   int(cfg_radar["num_rx"])
    chirp_loops     =   int(cfg_radar["num_doppler"])
    adc_samples     =   int(cfg_radar["num_range"])
    num_antennas    =   num_tx * num_rx


    if r_idxs.ndim != 1 or r_idxs.size == 0:
        raise ValueError("cfg_radar['range_idx'] must be a non-empty 1-D array.")
    
    if r_idxs.min() < 0:
        raise ValueError("cfg_radar['range_idx'] contains negative indices.")
    
    if r_idxs.max() >= adc_samples:
        raise ValueError(f"cfg_radar['range_idx'] contains index {r_idxs.max()}, but num_range={adc_samples}.")


    # State for frame differencing and (Windows Implementation) temporal averaging.
    last_frame = np.zeros((num_antennas, chirp_loops, adc_samples), dtype=np.complex64)
    history_len = max(2, int(cfg_cfar.get("temporal_history_frames", 5)))
    last_frames = np.zeros((history_len, num_antennas, chirp_loops, len(r_idxs)), dtype=np.complex64)


    # Get the antenna positions
    x_locs, _, _ = get_ant_pos_2d(num_antennas, adc_samples, num_rx)
    x_locs = np.asarray(x_locs).reshape(-1)

    # Setup the DCA1000
    print(f"-- Starting producer for DCA1000 with ip {static_ip} and system ip {system_ip}")
    dca = DCA1000(config_port=config_port, data_port=data_port, static_ip=static_ip, system_ip=system_ip)
    print("-- DCA1000 initialized.")



    # Setup Real-time Raw Data Collection
    # -----------------------------------   (if flag figures in terminal call)

    save_raw_dt = bool(cfg_radar.get("save_raw_dt", False))     # NOTE :    think of this like an on/off switch 
    bin_file = None                                             #           -- decides whether we "trigger" the "save proceedure" or not

    if save_raw_dt:     #   NOTE :  this block sets up things so that we record and save the incoming data 
                        #           if 'save_raw_dt' field was entered in the terminal
        raw_dir = cfg_radar.get("raw_dir", exp_path)
        os.makedirs(raw_dir, exist_ok=True)
        bin_path = os.path.join(raw_dir, f"{exp_name}_raw.bin")

        if os.path.exists(bin_path):    # overwrite file with same name if it exists
            os.remove(bin_path)

            
        bin_file = open(bin_path, "ab")
        print(f"-- Saving raw data stream to {bin_path}")

    # --------------------------------------------------

    

    try:
        while not stop_event.is_set():
            raw = dca.read(
                timeout=float(cfg_cfar.get("read_timeout_s", 3.0)),
                chirps=chirp_loops,
                rx=num_rx,
                tx=num_tx,
                samples=adc_samples,
            )

            if stop_event.is_set():
                break
            if raw is None:
                continue

            if save_raw_dt and bin_file is not None:
                raw.astype(np.int16, copy=False).tofile(bin_file)

            if not q.empty():
                # Keep realtime behavior: drop frames instead of building latency.
                continue

            # DCA organize -> (chirp_loops * num_tx, num_rx, adc_samples)
            raw = dca.organize(raw, chirp_loops, num_tx, num_rx, adc_samples)

            # Apply Hamming window (range window)
            adc_windowed = raw * np.hamming(adc_samples)


            # Reshape the data to (num_tx*num_rx, chirp_loops, adc_samples)
            beat_freq_data = adc_windowed.reshape(chirp_loops, num_tx, num_rx, adc_samples)
            beat_freq_data = beat_freq_data.transpose(1, 2, 0, 3)
            beat_freq_data = beat_freq_data.reshape(num_antennas, chirp_loops, adc_samples)

            # Apply FFT along the range dimension
            range_fft = np.fft.fft(beat_freq_data, axis=-1)
            last_frame_fft = np.fft.fft(last_frame, axis=-1)

            # Update the last frame
            last_frame = beat_freq_data

            #   WINDOWS IMPLEMENTATION DEFAULT : background subtraction for non-Doppler mode;
            #   -- Doppler mode relies on the moving-velocity notch.


            #    IDEA : In non-Doppler mode, remove the previous frame from the current frame.
            #   
            #    Purpose:
            #      Static objects usually appear similarly from frame to frame.
            #      Moving objects change between frames.
            #   
            #    So:   current_frame - previous_frame
            #    suppresses static clutter and keeps mostly motion/change
            #   
            #    Default behavior:
            #      - if doppler=False, bg_sub defaults to True
            #      - if doppler=True,  bg_sub defaults to False
            #   
            #    Why?
            #      In Doppler mode, moving/static separation is already handled by the
            #      Doppler/velocity filtering, so background subtraction is less necessary.

            if bool(cfg_cfar.get("bg_sub", not bool(cfg_radar.get("doppler", False)))):
                range_fft = range_fft - last_frame_fft


            # Keep only the selected range bins
            range_fft_s = range_fft[:, :, r_idxs]


            # Zero out the first few selected range bins

            #   IDEA :    the closest range bins are often polluted by hardware leakage, TX/RX
            #             coupling, DC artifacts, or reflections very close to the radar board.
            #
            #             near_range_zero_bins controls how many of these bins are forced to zero.
            #
            #             min(...) protects against asking to zero more bins than exist.

            near_zero = min(int(cfg_cfar.get("near_range_zero_bins", 5)), range_fft_s.shape[-1])
                        
            if near_zero > 0:
                range_fft_s[:, :, :near_zero] = 0


            #   optionally zero out the last few selected range bins
            #
            #   IDEA :  the farthest bins can sometimes be noisy or outside the useful detection region. 
            #           By default this is 0, meaning no far bins are removed
            #
            far_zero = min(int(cfg_cfar.get("far_range_zero_bins", 0)), range_fft_s.shape[-1])  # NOTE : controls how many bins at the end are forced to zero
            if far_zero > 0:
                range_fft_s[:, :, -far_zero:] = 0


            #   UPDATE ROLLING FRAME BUFFER
            #   ---------------------------
            #   NOTE : 'last_frames' stores the most recent processed range-FFT frames
            #
            last_frames[:-1] = last_frames[1:]      # shifts older frames one position toward the beginning
            last_frames[-1] = range_fft_s           # store the current cleaned frame as the newest frame


            if bool(cfg_radar.get("doppler", False)):

                current = last_frames[-1]                       # (num_ant, chirp_loops, range_bins)
                n_chirps = current.shape[1]

                # Doppler FFT across chirp_loops axis.
                doppler = np.fft.fftshift(np.fft.fft(current, n=n_chirps, axis=1), axes=1)

                # Zero-velocity notch: kill bins near DC (the static pipe)
                mid = n_chirps // 2         # bin 16 == zero velocity
                n_notch = 1 #min(max(0, int(cfg_cfar.get("doppler_notch_bins", 1))), max(0, mid - 1))

                if n_notch > 0:
                    doppler[:, mid - n_notch: mid + n_notch + 1, :] = 0

                # Select the strongest moving velocity bin per range, but only among velocity bins passing the adaptive SNR gate

                power = np.sum(np.abs(doppler), axis=0)         # (doppler, range)        
                                                                #    NOTE : Coherent across antennas: pick the strongest moving velocity
                                                                #           bin per range using power summed across antennas 
                                                                #           (same velocity bin for all antennas at each range,
                                                                #           so cross-antenna phase is preserved for beamforming).

                energy = np.sum(np.abs(doppler) ** 2, axis=0)   # (doppler, range)

                # SNR THRESHOLD
                snr = energy / (np.median(energy) + 1e-6)
                valid = snr > float(cfg_cfar.get("doppler_snr_threshold", 2.0))

                masked_power = np.where(valid, power, -np.inf)
                has_valid = np.any(valid, axis=0)
                best_vel_valid = np.argmax(masked_power, axis=0)
                best_vel_fallback = np.argmax(power, axis=0)
                best_vel = np.where(has_valid, best_vel_valid, best_vel_fallback)

                r_axis = np.arange(doppler.shape[2])
                bf_input = doppler[:, best_vel, r_axis]           # (num_ant, range_bins) COMPLEX


                #   NOISE MASK  : only keep range bins whose peak moving-power exceeds an
                #                 adaptive noise floor. Without this, every empty range bin still picks
                #                 some argmax velocity (just noise) and gets beamformed to a random angle,
                #                 producing scattered speckle across the whole heatmap.

                peak_power = power[best_vel, r_axis]                                                        # (range_bins,)
                noise_floor = np.median(peak_power) * float(cfg_cfar.get("doppler_noise_multiplier", 5.0))  # 5x median (tune: 2–5)//each step up kills 0.14m/s of velocity
                moving_mask = (peak_power > noise_floor) & has_valid                                        # (range_bins,) bool
                bf_input = bf_input * moving_mask[np.newaxis, :]

            else:
                # WINDOWS IMPLEMENTATION non-Doppler mode :         NOTE : with  bg_sub on (kills static clutter)
                #       -> use the latest bg-subtracted frame,
                #          with a short coherent average for SNR but no long motion smear.
                avg_n = min(max(1, int(cfg_cfar.get("dense_average_frames", 2))), history_len)
                bf_cube = np.mean(last_frames[-avg_n:], axis=0)   # (ant, chirp, range)
                bf_input = np.mean(bf_cube, axis=1)               # (ant, range)

            bf_output = beamform_2d(bf_input, cfg_radar, x_locs)

            max_output = float(np.max(np.abs(bf_output)))
            if not np.isfinite(max_output) or max_output <= 0.0:
                max_output = 1.0

            if bool(cfg_cfar.get("cfar_on", False)):
                bf_output = process_frame_2d(np.abs(bf_output) ** 2, cfg_cfar) / max_output
            else:
                bf_output = bf_output / max_output

            try:
                q.put_nowait(("bev", bf_output.astype(np.float32, copy=False)))
            except queue.Full:
                continue

    except KeyboardInterrupt:
        print(f"-- Producer for DCA1000 with ip {static_ip} stopped by user.")

    finally:
        if bin_file is not None:
            bin_file.close()
            print(f"-- Raw data file for {exp_name} has been safely closed.")
        try:
            dca.close()
        except Exception:
            pass