from . import realtime_streaming_1r
import numpy as np


def main():
    """
    Main function to start the real-time radar streaming and processing.
    """

    # Parameters for the range-azimuth beamforming
    r_idxs = np.arange(0, 100, 1)
    phi = np.deg2rad(np.arange(0, 180, 1))
    
    width = 40 # azimuth width in degrees

    # Radar  parameters
    cfg_radar = {
        "range_idx": r_idxs,
        "phi": phi,
        "width": width,
        "num_tx": 3,
        "num_rx": 4,
        "num_doppler": 16,
        "num_range": 992,
        "sample_rate": 5166000,
        "c": 3e8,
        "lm": 3e8 / 77e9,
        "slope": 70.150e6
    }

    # Parameters for CFAR
    cfg_cfar = {
        "num_train_r": 10,
        "num_train_d": 8,
        "num_guard_r": 2,
        "num_guard_d": 2,
        "threshold_scale": 1e-3
    }

    print("⌛️ Starting streaming...")

    # Start the streaming process
    realtime_streaming_1r.main(cfg_radar, cfg_cfar)

if __name__ == "__main__":
    main()