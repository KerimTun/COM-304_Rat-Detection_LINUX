import argparse
from pathlib import Path
from datetime import datetime
import json
import os

from . import realtime_streaming_1r

from . import configure

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


#current_dir = (os.path.dirname(os.getcwd())) # one level up for this repo


def sanitize_filename_part(name):
    """
    Return a filesystem-safe experiment name component.
    """

    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))
    return safe.strip("_") or "experiment"


def build_experiment_paths(exp_name):
    """
    Create one directory tree for the whole live experiment.

    Layout:
        Data_Live_Experiments/<exp_name>_<timestamp>/
            experiment_metadata.json
            raw/
                radar1_raw.bin
            cartesian_frames/
                metadata.json
                index.jsonl
                x_axis.npy
                y_axis.npy
                frames/
                    frame_000000.npz
                    frame_000001.npz
                    ...
            mp4/
                cartesian.mp4
    """

    safe_exp_name   = sanitize_filename_part(exp_name)
    timestamp       = datetime.now().strftime("%Y-%m-%d_%H")
    recording_name  = f"{safe_exp_name}_{timestamp}"

    experiment_dir  = PROJECT_ROOT / "Data_Live_Experiments" / recording_name
    raw_dir         = experiment_dir / "raw"
    cartesian_dir   = experiment_dir / "cartesian_frames"
    mp4_dir         = experiment_dir / "mp4"

    for directory in (experiment_dir, raw_dir, cartesian_dir, mp4_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "project_root"          : str(PROJECT_ROOT),
        "experiment_timestamp"  : timestamp,
        "recording_name"        : recording_name,
        "experiment_dir"        : str(experiment_dir),
        "raw_dir"               : str(raw_dir),
        "cartesian_dir"         : str(cartesian_dir),
        "mp4_dir"               : str(mp4_dir),
    }



def make_beamforming_angles(width_deg=40.0, center_deg=90.0):
    """
    Create the beamforming angular grid.

    Beamformer convention:
        phi = 90° means straight ahead / broadside.

    Cartesian convention:
        azimuth = 0° means straight ahead.

    Therefore:
        azimuth_deg = phi_deg - 90
    """

    phi_deg = np.arange(center_deg - width_deg / 2.0, center_deg + width_deg / 2.0 + 1, 1) # "angular" step size

    phi = np.deg2rad(phi_deg)
    azimuth_deg = phi_deg - 90.0

    return phi, phi_deg, azimuth_deg




def make_radar_poses(y_mid_gap, yaw1_deg=0.0, yaw2_deg=0.0):
    """
    Define the two radar poses in a shared global coordinate frame.

    The global coordinate frame is centered at the midpoint between the two
    radars.

    Coordinate convention:

        +x = forward
        +y = left

    Therefore, if the two radars are separated laterally by `y_mid_gap`,
    their default positions are:

        Radar 1: y = +y_mid_gap / 2    -- on the left side
        Radar 2: y = -y_mid_gap / 2    -- on the right side

    Parameters
    ----------
    y_mid_gap : float
        Distance between the two radars in meters.

    yaw1_deg : float, optional
        Yaw angle of Radar 1 in degrees.

        Default is 0 degrees.

    yaw2_deg : float, optional
        Yaw angle of Radar 2 in degrees.

        Default is 0 degrees.

    Returns
    -------
    tuple
        radar1_pose, radar2_pose

        Each pose has the form:

            (x, y, yaw_deg)

    Example
    -------
    For two radars separated by 0.8 m:

        radar1_pose = (0.0, +0.4, 0.0)
        radar2_pose = (0.0, -0.4, 0.0)
    """

    radar1_pose = (0.0,  y_mid_gap / 2.0, yaw1_deg)    # Left  side of origin (on Y-axis)
    radar2_pose = (0.0, -y_mid_gap / 2.0, yaw2_deg)    # Right side of origin (on Y-axis)

    return radar1_pose, radar2_pose


def make_radar_pose(yaw1_deg=0.0):
    """
    Define the two radar poses in a shared global coordinate frame.

    The global coordinate frame is centered at the midpoint between the two
    radars.

    Coordinate convention:

        +x = forward
        +y = left

    Parameters
    ----------

    yaw1_deg : float, optional
        Yaw angle of Radar 1 in degrees.

        Default is 0 degrees.


    Returns
    -------
    tuple
        radar1_pose

        Single pose has the form:

            (x, y, yaw_deg)

    Example
    -------
    For single radars separated by 0.8 m:

        radar1_pose = (0.0, +0.4, 0.0)
    """

    radar1_pose = (0.0,  0.0, yaw1_deg)    # Left side of origin (on Y-axis)

    return radar1_pose




def build_cfg_radar(cfg_params, mid_gap_m=0.0, beam_width_deg=40.0, beam_center_deg=90.0, exp_name="test", save_raw_dt=False, doppler=False):
    """
    Build the radar configuration dictionary used by the streaming pipeline.
    
    This function converts the parameters computed from the TI `.cfg` file into
    the format expected by the real-time two-radar pipeline.

    It defines:
        - the full range-bin grid
        - the beamforming azimuth grid
        - the two radar poses
        - radar dimensions from the `.cfg`
        - Cartesian grid limits for fusion / CNN input

    Parameters
    ----------
    cfg_params : dict
        Parameters computed from `configure.compute_params()`.

    mid_gap_m : float, optional
        Distance between the two radars in meters.
        Default is 0.3.

    beam_width_deg : float, optional
        Angular width scanned by the beamformer, in degrees.
        Default is 40.

    beam_center_deg : float, optional
        Center angle of the beamforming sector, in degrees.
        With the current convention, 90° means straight ahead.
        Default is 90.

    Returns
    -------
    dict
        Runtime radar configuration dictionary used by:
            - prod_dca.py
            - realtime_streaming_2r_fused.py
            - beamform_2d_s()
            - Cartesian projection / fusion code

    Notes
    -----
    Angle convention:
        phi_deg is used for beamforming.
        phi = 90° means straight ahead.

        azimuth_deg is used for Cartesian projection.
        azimuth = 0° means straight ahead.

    Range convention:
        num_range is the full number of ADC/range FFT bins.
        range_idx contains the selected range-bin indices.
        In this simplified version, all range bins are selected.
    """
        
    # Range selection: use all range bins
    # -----------------------------------
    num_range_bins = int(cfg_params["NUM_RANGE_BINS"])
    range_resolution = cfg_params["RANGE_RESOLUTION"]
    r_idxs = np.arange(num_range_bins, dtype=int)
    range_bins_m = r_idxs * range_resolution        # range bins with meter-valued entries (w.r.t. range resolution)


    # Beamforming angles
    # ------------------
    phi, phi_deg, azimuth_deg = make_beamforming_angles(width_deg=beam_width_deg, center_deg=beam_center_deg )

    # Radar pose
    # -----------
    radar1_pose = (0.0,  0.0, 0.0)
    #make_radar_poses(y_mid_gap=mid_gap_m, yaw1_deg=0.0, yaw2_deg=0.0)


    # Cartesian grid parameters
    # -------------------------
    cart_res_m = range_resolution   # Grid resolution (spacing between each pixel) = range_resolution of our radar
    padding_y = mid_gap_m/2        # radar offset to pad sides of the grid (since otherwise each radar's max range exceeds the unpadded grid sides) 

    cart_x_min_m = 0.0                      # X-axis : forward direction; 0 starts at radar midpoint
    cart_x_max_m = cfg_params["MAX_RANGE"]
    cart_y_min_m = -cfg_params["MAX_RANGE"] - padding_y  # y: right side + radar offset
    cart_y_max_m =  cfg_params["MAX_RANGE"] + padding_y  # y: left side + radar offset

    # Frequency / units
    # -----------------
    start_freq_hz = cfg_params["START_FREQ"] * 1e9

    sample_rate_ksps = cfg_params["SAMPLE_RATE"]
    sample_rate_hz = sample_rate_ksps * 1e3

    slope_mhz_per_us = cfg_params["FREQ_SLOPE"]
    slope_hz_per_s = slope_mhz_per_us * 1e12

    # Experiment output directory tree
    experiment_paths = build_experiment_paths(exp_name)

    # Final config
    # -------------

    cfg_radar = {
        # Physical constants
        "c"     : 3e8,                                           # Speed of light [m/s]
        "lm"    : 3e8 / start_freq_hz,                           # Radar wavelength [m]

        # Range information
        "range_idx"          : r_idxs,                           # All range FFT bin indices
        "range_bins_m"       : range_bins_m,                     # Range bins converted to meters
        "range_resol"        : range_resolution,                 # Range resolution [m/bin]
        "max_range"          : cfg_params["MAX_RANGE"],          # Max unambiguous range [m]

        # Beamforming angle grid
        "phi"                : phi,                              # Beamforming angles [rad]
        "phi_deg"            : phi_deg,                          # Beamforming angles [deg]
        "azimuth_deg"        : azimuth_deg,                      # Physical azimuth [deg]
        "phi_width_deg"      : beam_width_deg,                   # Beamforming sector width [deg]
        "phi_center_deg"     : beam_center_deg,                  # Beamforming center angle [deg]
        "phi_convention"     : "beamformer_phi_center_90_deg",   # phi=90° means forward

        # Single-radar geometry
        "n_radar"            : 1,                                # Number of radars
        "dist_between_radars": mid_gap_m,                        # Radar mid_gap [m]
        "radar1_pose"        : radar1_pose,                      # Radar 1 pose: (x, y, yaw_deg)

        # Radar dimensions from .cfg
        "num_tx"             : int(cfg_params["NUM_TX"]),           # Number of active TX antennas
        "num_rx"             : int(cfg_params["NUM_RX"]),           # Number of active RX antennas
        "num_doppler"        : int(cfg_params["NUM_DOPPLER_BINS"]), # Doppler bins, old key
        "num_doppler_bins"   : int(cfg_params["NUM_DOPPLER_BINS"]), # Doppler bins, explicit key
        "num_range"          : num_range_bins,                      # Full ADC samples / range FFT bins

        # Doppler parameters
        "max_doppler"        : cfg_params["MAX_DOPPLER"],         # Max radial velocity [m/s]
        "doppler_resol"      : cfg_params["DOPPLER_RESOLUTION"],  # Doppler resolution [m/s/bin]

        # Original units from .cfg
        "sample_rate_ksps"   : sample_rate_ksps,         # ADC sample rate [ksps]
        "slope_mhz_per_us"   : slope_mhz_per_us,         # Chirp slope [MHz/us]

        # SI units
        "sample_rate_hz"     : sample_rate_hz,           # ADC sample rate [Hz]
        "slope_hz_per_s"     : slope_hz_per_s,           # Chirp slope [Hz/s]

        # Backward-compatible aliases
        "sample_rate"        : sample_rate_ksps,         # Alias for old code, unit: ksps
        "slope"              : slope_mhz_per_us,         # Alias for old code, unit: MHz/us

        # Cartesian grid parameters
        "cart_x_min_m"       : cart_x_min_m,             # Cartesian grid min x [m]  -- NOTE : minimum is set to the origin
        "cart_x_max_m"       : cart_x_max_m,             # Cartesian grid max x [m]  -- NOTE : max range of radar
        "cart_y_min_m"       : cart_y_min_m,             # Cartesian grid min y [m]  -- NOTE : padded max range on the right side (since origin is at midpoint between radars)
        "cart_y_max_m"       : cart_y_max_m,             # Cartesian grid max y [m]  -- NOTE : padded max range on the left side (since origin is at midpoint between radars)
        "cart_res_m"         : cart_res_m,               # Cartesian grid resolution [m/pixel]


        # Experiment Recording 
        # "save_raw_dt"       : save_raw_dt,
        # "exp_name"          : exp_name,
        # "exp_path"          : os.path.join(current_dir, "data"),   # NOTE : Data Directory Path (for specified experiment file)
        # "doppler"           : doppler                              # Boolean Value -- Indicates wether running experiment is set to operate with/out doppler


        # Experiment recording
        "save_raw_dt"       : save_raw_dt,
        "exp_name"          : exp_name,
        "exp_path"          : experiment_paths["raw_dir"],  # backward-compatible alias -- # NOTE : Data Directory Path (for specified experiment file)
        "doppler"           : doppler,                       # Boolean Value -- Indicates wether running experiment is set to operate with/out doppler

        # Unified output tree for this live experiment
        **experiment_paths,


    }

    return cfg_radar





def build_cfg_cfar(cfar_on=False):
    """
    Build the CFAR configuration dictionary.

    CFAR stands for Constant False Alarm Rate. It is used to detect targets by
    comparing each cell against a local estimate of the surrounding noise floor.

    Returns
    -------
    dict
        CFAR configuration dictionary.

    Notes
    -----
    The exact interpretation of `threshold_scale` depends on the implementation
    of the CFAR function used downstream.
    """

    return {
        "cfar_on"    : cfar_on,     
        "num_train_r": 10,          # Number of training cells in the range direction
        "num_train_d": 8,           # Number of training cells in the Doppler direction.
        "num_guard_r": 2,           # Number of guard cells in the range direction
        "num_guard_d": 2,           # Number of guard cells in the Doppler direction
        "threshold_scale": 1e-3,    # Detection threshold scaling factor
    }



def print_runtime_summary(cfg_file_path, cfg_radar, cfg_cfar):
    """
    Print the selected runtime parameters before streaming starts.
    """

    print()
    #print("----------------------------------------------------")
    print("--------- RUNTIME CONFIGURATION ----------")
    print(f"  Config file:  {cfg_file_path}")

    print()
    print("   -- Range :")
    print(f"  Range bins:             {cfg_radar['range_idx'][0]} to {cfg_radar['range_idx'][-1]}")
    print(f"  Range meters:           {cfg_radar['range_bins_m'][0]:.2f}m to {cfg_radar['range_bins_m'][-1]:.2f}m")
    print(f"  Num range bins:         {len(cfg_radar['range_idx'])}")
    print(f"  Range resolution:       {cfg_radar['range_resol']:.4f}m")

    print()
    print("   -- Beamforming :")
    print(f"  Beamformer phi range:   {cfg_radar['phi_deg'][0]:.1f}° to {cfg_radar['phi_deg'][-1]:.1f}°")
    print(f"  Physical azimuth range: {cfg_radar['azimuth_deg'][0]:.1f}° to {cfg_radar['azimuth_deg'][-1]:.1f}°")
    print(f"  Num phi bins:           {len(cfg_radar['phi'])}")

    print()
    print("   -- Radar geometry :")
    #print(f"  mid_gap:               {cfg_radar['dist_between_radars']:.3f} m")
    print(f"  Radar 1 pose:           {cfg_radar['radar1_pose']}")
    #print(f"  Radar 2 pose:           {cfg_radar['radar2_pose']}")

    print()
    print("   -- Radar dimensions :")
    print(f"  num_tx:                 {cfg_radar['num_tx']}")
    print(f"  num_rx:                 {cfg_radar['num_rx']}")
    print(f"  num_doppler:            {cfg_radar['num_doppler']}")
    print(f"  num_range:              {cfg_radar['num_range']}")

    print()
    print("   -- Cartesian fusion grid :")
    print(f"  x:                      {cfg_radar['cart_x_min_m']:.2f}m to {cfg_radar['cart_x_max_m']:.2f}m")
    print(f"  y:                      {cfg_radar['cart_y_min_m']:.2f}m to {cfg_radar['cart_y_max_m']:.2f}m")
    print(f"  resolution:             {cfg_radar['cart_res_m']:.4f}m")
    #print("----------------------------------------------------")
    print()

    if cfg_radar["save_raw_dt"]:
        print(f" Recording Acquired Data from Radars") #{cfg_radar["save_raw_dt"]}")

    # print(f"  Experiment name :       {cfg_radar["exp_name"]}_[Radar1/2]")   
    # print(f"  Path Name :             {cfg_radar["exp_path"]}_[Radar1/2]") 


    print(f"  Experiment name :       {cfg_radar['recording_name']}")
    print(f"  Experiment dir :        {cfg_radar['experiment_dir']}")
    print(f"  Raw data dir :          {cfg_radar['raw_dir']}")
    print(f"  Cartesian frames dir :  {cfg_radar['cartesian_dir']}")
    print(f"  MP4 dir :               {cfg_radar['mp4_dir']}")

    if cfg_radar["doppler"]:
        print(f"  Doppler Mode :      ON ")    
    else:
        print(f"  Doppler Mode :      OFF")    

    if cfg_cfar["cfar_on"]:
        print(f"  CFAR :              ON")
    else:
        print(f"  CFAR :              OFF")




def main():


    #   PARSER --------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Example script with command line arguments.")

    parser.add_argument("--config", default="my_config", help="Name of the .cfg file inside the configs directory.")

    # Add arguments
    parser.add_argument("--cfar"        , action="store_true"         , help="True if you want cfar.")
    parser.add_argument("--doppler"     , action="store_true"         , help="True if you want doppler.")
    parser.add_argument("--save_raw_dt" , action="store_true"         , help="True if you want to save the real-time captured raw data to 'data/<exp_name>_Raw_0.bin'.")
    parser.add_argument("--exp_name"    , type=str   , default="test" , help="Base filename for saved raw data")
    # parser.add_argument("--mid_gap"    , type=float , default=0.0   , help="Distance between the two radars in meters.")
    parser.add_argument("--beam-width"  , type=float , default=180.0  , help="Beamforming angular width in degrees.")
    parser.add_argument("--beam-center" , type=float , default=90.0   , help="Beamformer center angle in degrees. 90° means straight ahead.")

    args = parser.parse_args()
    #   ---------------------------------------------------------------------------------------------------------------


    # -------  Process Config File

    cfg_file_path = f"../configs/{args.config}.cfg"  # NOTE : THIS IS THE PATH TO THE CONFIG FILE (equivalent to lua files)

    # Parse .cfg file
    cfg = configure.parse_cfg(cfg_file_path)

    # Compute radar parameters from .cfg
    cfg_params = configure.compute_params(cfg)

    # Print computed radar parameters
    configure.print_params(cfg_params)


    # -------  Build runtime configs

    cfg_radar = build_cfg_radar(cfg_params, mid_gap_m=0.0, beam_width_deg=args.beam_width, beam_center_deg=args.beam_center, exp_name=args.exp_name, save_raw_dt=args.save_raw_dt, doppler=args.doppler)
    cfg_cfar  = build_cfg_cfar(args.cfar)

    # Print runtime summary
    # print_runtime_summary(cfg_file_path, cfg_radar)

    print_runtime_summary(cfg_file_path, cfg_radar, cfg_cfar)

    experiment_metadata_path = Path(cfg_radar["experiment_dir"]) / "experiment_metadata.json"

    with open(experiment_metadata_path, "w") as f:
        json.dump(
            {
                "config_file": cfg_file_path,
                "cfg_params": cfg_params,
                "cfg_radar": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in cfg_radar.items()
                },
                "cfg_cfar": cfg_cfar,
            },
            f,
            indent=4,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )

    print()
    print("--------- STARTING STREAMING ----------")
    realtime_streaming_1r.main(cfg_radar, cfg_cfar)
    print("---------------------------------------")
    print()





if __name__ == "__main__":
    main()

