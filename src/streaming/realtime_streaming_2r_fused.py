"""
Dual-radar realtime streaming/fusion pipeline.
    - two radar producers run as child processes;
    - the visualizer/recorder runs as a child consumer process;
    - the main process owns the shared multiprocessing.Event and joins/terminates
      children in the same order as project_3 implementation.
"""

import os
import sys
import time
import json
import warnings
import signal
from collections import deque
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Queue, Event

warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import numpy as np

from panda3d.core import loadPrcFileData

# Panda3D is only used as a task loop here, not for rendering/audio.
# Disable its window and audio system before importing ShowBase.
loadPrcFileData("", "window-type none")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "audio-music-active #f")
loadPrcFileData("", "audio-sfx-active #f")

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

import matplotlib
matplotlib.use("Qt5Agg")

# Prevent plot windows from jumping above everything else.
matplotlib.rcParams["figure.raise_window"] = False

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-dark")

from PyQt5 import QtWidgets

import imageio.v2 as imageio

from .prod_dca import producer_real_time_1843_SAVE_DOPPLER

from visualization.visualization import configure_ax_bf
from utils.utils import radar_power_on_cartesian_grid, normalize_for_display



def _as_jsonable(value):
    """
    Convert common NumPy / pathlib values into JSON-serializable Python values.

    Used when saving metadata.json, because json.dump() cannot directly handle
    NumPy arrays, NumPy scalar types, or pathlib.Path objects.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value




class JerryClassifier:
    """
    Range-bin activity detector.

    It tracks whether a configurable range-bin ROI is active over a moving
    window of frames. Each radar gets its own instance in the dual-radar
    visualizer.
    """

    def __init__(self, range_bins, sensitivity=1.0, min_active_bins=1, frame_window=10, num_frames_thresh=0.25 ):
        """Initialize a per-radar range-bin activity detector.
        
        Args:
            range_bins: Iterable of range-bin indices that define the detection ROI.
            sensitivity: Multiplier applied to the estimated noise floor to create
                the per-bin activity threshold.
            min_active_bins: Minimum number of ROI bins that must exceed the
                threshold for the current frame to be considered active.
            frame_window: Number of recent frame-level active/inactive decisions used
                to smooth detection.
            num_frames_thresh: Minimum fraction of active frames inside the moving
                window required to return a positive detection.
        
        Side effects:
            Stores detector configuration and creates ``self.active_frames``, a fixed
            length queue used to smooth detections over time.
        """
        self.range_bins = np.asarray(range_bins, dtype=int)
        self.sensitivity = float(sensitivity)
        self.min_active_bins = int(min_active_bins)
        self.active_frames = deque(maxlen=int(frame_window))
        self.num_frames_thresh = float(num_frames_thresh)

    def update_detection(self, bf_output_1d):
        """Update the range-bin activity detector with one 1D radar profile.
        
        The input is expected to be a single-radar range profile, typically obtained
        from a beamformed polar map using ``np.abs(bf).max(axis=0)``. The method
        compares the configured ROI against a noise floor estimated from bins outside
        the ROI, then smooths the frame-level activity decision over a moving window.
        
        Args:
            bf_output_1d: One-dimensional array of signal amplitudes indexed by range
                bin.
        
        Returns:
            tuple:
                detected: True when the moving-window activity rate exceeds
                    ``num_frames_thresh``.
                detection_rate: Fraction of recent frames that were active.
                active_bins: Number of ROI bins above the current threshold.
                noise_floor: Mean non-zero amplitude outside the ROI.
                bin_threshold: Activity threshold used for this frame.
                roi_max: Maximum amplitude inside the ROI.
        """
        bf_output_1d = np.abs(np.asarray(bf_output_1d, dtype=np.float32))
        if bf_output_1d.size == 0:
            self.active_frames.append(0)
            return False, 0.0, 0, 0.0, 0.0, 0.0

        roi_bins = self.range_bins[(self.range_bins >= 0) & (self.range_bins < bf_output_1d.size)]
        if roi_bins.size == 0:
            roi_bins = np.arange(bf_output_1d.size)

        outside_mask = np.ones(bf_output_1d.size, dtype=bool)
        outside_mask[roi_bins] = False
        outside_roi = bf_output_1d[outside_mask]
        nonzero_vals = outside_roi[outside_roi > 0]
        noise_floor = float(np.mean(nonzero_vals)) if nonzero_vals.size > 0 else 1.0
        bin_threshold = self.sensitivity * noise_floor

        roi = bf_output_1d[roi_bins]
        active_bins = int(np.sum(roi > bin_threshold))
        active_frame_flag = 1 if active_bins >= self.min_active_bins else 0

        self.active_frames.append(active_frame_flag)
        detection_rate = float(np.mean(self.active_frames)) if self.active_frames else 0.0
        detected = detection_rate >= self.num_frames_thresh
        roi_max = float(np.max(roi)) if roi.size > 0 else 0.0

        return detected, detection_rate, active_bins, noise_floor, bin_threshold, roi_max


class CartesianEmaDetector:
    """
    Exponential Moving Average (EMA) detector on the fused Cartesian global map.

    By default the ROI is the whole global Cartesian map. If cfg_radar contains
    ``detector_cart_roi_m`` with x/y bounds, the detector only evaluates that
    rectangular global ROI.
    """

    def __init__(self, x_grid, y_grid, roi=None, threshold=0.35, ema_alpha=0.5, marker_min_amp=0.25):
        """Initialize a diagnostic detector on the fused Cartesian map.
        
        This detector is useful for localization, marker placement, and debugging the
        global map. It should generally not be the only source of the final detection
        decision, because Cartesian fusion and interpolation can change signal
        statistics.
        
        Args:
            x_grid: Cartesian x-coordinate mesh, in meters.
            y_grid: Cartesian y-coordinate mesh, in meters.
            roi: Optional dictionary with ``x_min``, ``x_max``, ``y_min``, and
                ``y_max`` bounds in meters. If omitted, the full map is used.
            threshold: EMA threshold above which the Cartesian ROI is considered active.
            ema_alpha: Weight given to the previous EMA value. Higher values smooth
                more aggressively.
            marker_min_amp: Minimum normalized ROI amplitude required before returning
                a peak marker.
        
        Side effects:
            Builds ``self.mask``, the boolean Cartesian ROI mask, and initializes the
            ROI EMA state.
        """
        self.threshold = float(threshold)
        self.ema_alpha = float(ema_alpha)
        self.marker_min_amp = float(marker_min_amp)
        self.roi_ema = 0.0

        if roi:
            x_min = float(roi.get("x_min", np.min(x_grid)))
            x_max = float(roi.get("x_max", np.max(x_grid)))
            y_min = float(roi.get("y_min", np.min(y_grid)))
            y_max = float(roi.get("y_max", np.max(y_grid)))
            self.mask = (x_grid >= x_min) & (x_grid <= x_max) & (y_grid >= y_min) & (y_grid <= y_max)
        else:
            self.mask = np.ones_like(x_grid, dtype=bool)

    def update(self, cartesian_power, x_grid, y_grid):
        """Update the Cartesian EMA detector with one fused Cartesian power map.
        
        The map is internally normalized to its own maximum before ROI statistics are
        computed. This makes the detector relative to the current frame, which is useful
        for display/localization but less reliable than per-radar domain detection for
        final target decisions.
        
        Args:
            cartesian_power: Two-dimensional fused Cartesian power map.
            x_grid: Cartesian x-coordinate mesh matching ``cartesian_power``.
            y_grid: Cartesian y-coordinate mesh matching ``cartesian_power``.
        
        Returns:
            tuple:
                detected: True if the ROI EMA is above ``self.threshold``.
                roi_ema: Current exponential moving average of ROI activity.
                roi_max: Current maximum normalized value inside the ROI.
                peak: ``None`` or a dictionary containing the strongest point as
                    ``x_m``, ``y_m``, and normalized ``value``.
        """
        z = np.abs(np.asarray(cartesian_power, dtype=np.float32))
        zmax = float(np.max(z)) if z.size else 0.0
        z_norm = z / zmax if zmax > 0 else z

        roi_vals = z_norm[self.mask]
        roi_max = float(np.max(roi_vals)) if roi_vals.size else 0.0
        self.roi_ema = self.ema_alpha * self.roi_ema + (1.0 - self.ema_alpha) * roi_max
        detected = self.roi_ema > self.threshold

        peak = None
        if detected and roi_max > self.marker_min_amp:
            masked = np.where(self.mask, z_norm, 0.0)
            flat_idx = int(np.argmax(masked))
            iy, ix = np.unravel_index(flat_idx, z_norm.shape)
            peak = {
                "x_m": float(x_grid[iy, ix]),
                "y_m": float(y_grid[iy, ix]),
                "value": float(z_norm[iy, ix]),
            }

        return detected, float(self.roi_ema), roi_max, peak




class MyApp(ShowBase):

    def __init__(self, queue_1, queue_2, cfg_radar, cfg_cfar, stop_event):
        """Create the dual-radar realtime visualizer/recorder consumer.
        
        This constructor runs inside the consumer process. It stores queue/configuration
        references, creates output folders, initializes dataset and MP4 recording,
        builds the global Cartesian grid, creates per-radar and Cartesian detectors,
        and sets up all Matplotlib figures. It also registers the project_3-compatible
        window-close/Escape shutdown path and adds the Panda3D update task.
        
        Args:
            queue_1: Multiprocessing queue receiving radar-1 beamformed frames.
            queue_2: Multiprocessing queue receiving radar-2 beamformed frames.
            cfg_radar: Radar, Cartesian grid, recording, and detector configuration.
            cfg_cfar: CFAR/config dictionary saved into metadata for traceability.
            stop_event: Shared multiprocessing event used to stop all processes.


        Proceedure :

             1. Initialize ShowBase / Panda3D task loop
             2. Store queues, configs, stop_event, and runtime state
             3. Load radar geometry from cfg_radar
             4. Read recording options
             5. Create dataset / video / log folders
             6. Open MP4 writer
             7. Build the global Cartesian grid
             8. Save dataset metadata
             9. Create per-radar Jerry detectors
            10. Create optional Cartesian EMA detector
            11. Create radar 1 polar plot
            12. Create radar 2 polar plot
            13. Create fused Cartesian global-map plot
            14. Create detection-status plot
            15. Connect window-close and Escape to shutdown
            16. Register updateTask as the realtime loop

        """
        ShowBase.__init__(self)



        #   Store multiprocessing inputs and internal runtime state
        #   -------------------------------------------------------

        self.q1 = queue_1               # q1, q2 : receive already-beamformed frames from the two radar producer processes
        self.q2 = queue_2

        self.stop_event = stop_event    # shared between all processes -- Purpose : stops producers + consumer together

        self.cfg_radar = cfg_radar
        self.cfg_cfar = cfg_cfar

        self.latest_msg = {}            # stores the newest frame from each radar
        self.msg_count = set()          # tracks whether both radars have produced a new synchronized pair

        # Shutdown / recording book-keeping
        self.is_closing = False
        self.closed = False
        self.video_writer = None
        self.frame_pair_idx = 0


        #   Load geometry / radar-axis information from the config
        #   ------------------------------------------------------

        self.phi            = np.asarray(cfg_radar["phi"])
        self.azimuth_deg    = np.asarray(cfg_radar["azimuth_deg"])
        self.range_bins_m   = np.asarray(cfg_radar["range_bins_m"], dtype=np.float32)

        # these tell the Cartesian mapper where each radar is located in the global map
        self.radar1_pose    = tuple(cfg_radar["radar1_pose"])
        self.radar2_pose    = tuple(cfg_radar["radar2_pose"])



        #   Read recording options
        #   ----------------------
        #
        self.record_dataset = bool(cfg_radar.get("record_dataset", True))       # controls .npz frame-pair saving
        self.record_mp4     = bool(cfg_radar.get("record_mp4", True))           # controls video export of the fused Cartesian plot
        
        # these enables skipping frames to reduce disk usage
        self.record_data_frame_every_n_pairs    = int(cfg_radar.get("record_data_frame_every_n_pairs", 1))
        self.record_video_every_n_pairs         = int(cfg_radar.get("record_video_every_n_pairs", 1))

        self.video_fps = int(cfg_radar.get("video_fps", 10))


        #   DATASET OUTPUT PATHS
        #   --------------------    Each synchronized radar pair can be saved under cartesian_dir/frames
        #
        self.experiment_dir = Path(cfg_radar["experiment_dir"])
        self.recording_name = cfg_radar["recording_name"]
        self.dataset_dir    = Path(cfg_radar["cartesian_dir"])
        self.frames_dir     = self.dataset_dir / "frames"

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.index_path  = self.dataset_dir / "index.jsonl"     # keeps a lightweight table of all saved frame files



        #   VIDEO OUTPUT PATHS  
        #   -------------------   MP4 contains the live fused Cartesian map, not the raw radar data
        #
        self.video_dir = Path(cfg_radar["mp4_dir"])
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.video_dir / "cartesian.mp4"


        #   LOGGING PATHS
        #   -------------
        self.log_dir = self.experiment_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.signal_log_path    = self.log_dir / "signal_log.jsonl"     # records every processed synchronised dual radar frame pair
        self.detection_log_path = self.log_dir / "jerry_log.jsonl"      # records only frames where detection is positive



        #   Open the MP4 writer if video recording is enabled
        #   -------------------------------------------------
        #
        if self.record_mp4:
            self.video_writer = imageio.get_writer(self.video_path, fps=self.video_fps, codec="libx264", quality=8, macro_block_size=1)

        # NOTE : writer must be closed cleanly during shutdown, otherwise the MP4 file may be corrupted or missing its final metadata



        #   Cartesian global fusion grid
        #   ------------------------------------------------------
        #
        x = np.arange(
            cfg_radar["cart_x_min_m"],
            cfg_radar["cart_x_max_m"] + cfg_radar["cart_res_m"],
            cfg_radar["cart_res_m"],
        )

        y = np.arange(
            cfg_radar["cart_y_min_m"],
            cfg_radar["cart_y_max_m"] + cfg_radar["cart_res_m"],
            cfg_radar["cart_res_m"],
        )

        self.x_axis = x.astype(np.float32)   # x = forward
        self.y_axis = y.astype(np.float32)   # y = left/right

        self.X, self.Y = np.meshgrid(self.x_axis, self.y_axis)

        # Display orientation:  horizontal axis = y ;   vertical axis   = x
        self.cart_extent = [ self.y_axis.min(), self.y_axis.max(), self.x_axis.min(), self.x_axis.max() ]

        self.save_dataset_metadata()    # saves metadata for reproducibility
                                        # NOTE : writes grid axes, radar poses, config values, and CFAR settings
                                        #        before streaming begins, so the saved dataset can be interpreted later.



        #   INITIALIZING 2 per-radar JERRY DETECTORS
        #   ----------------------------------------    NOTE : each radar has its own JerryClassifier so their histories/noise floors are tracked separately
        #
        detector_bins = self._make_detector_bins()      # chooses the range-bin ROI

        self.detectors = [
            JerryClassifier(
                detector_bins,
                sensitivity=cfg_radar.get("detector_sensitivity", 1.0),
                min_active_bins=cfg_radar.get("detector_min_active_bins", 1),
                frame_window=cfg_radar.get("detector_frame_window", 10),
                num_frames_thresh=cfg_radar.get("detector_frame_threshold", 0.25),
            ),
            JerryClassifier(
                detector_bins,
                sensitivity=cfg_radar.get("detector_sensitivity", 1.0),
                min_active_bins=cfg_radar.get("detector_min_active_bins", 1),
                frame_window=cfg_radar.get("detector_frame_window", 10),
                num_frames_thresh=cfg_radar.get("detector_frame_threshold", 0.25),
            ),
        ]



        #   Initialize Cartesian EMA "detector" (only for visualisation purposes)
        #   ---------------------------------------------------------------------
        #
        self.cart_detector = CartesianEmaDetector(
            self.X,
            self.Y,
            roi=cfg_radar.get("detector_cart_roi_m"),
            threshold=cfg_radar.get("cart_detector_threshold", 0.35),
            ema_alpha=cfg_radar.get("cart_detector_ema_alpha", 0.5),
            marker_min_amp=cfg_radar.get("cart_detector_marker_min_amp", 0.25),
        )
        self.last_peak_artist = None        # stores the current plotted peak marker so it can be removed/replaced on the next frame




        #      PLOT SECTION
        #   ------------------
        #
        # --- RADAR 1 POLAR PLOT
        self.fig_1  = plt.figure(figsize=(6, 6))
        self.ax_1   = self.fig_1.add_subplot(111, projection="polar")
        self.im_1   = configure_ax_bf( self.ax_1, self.phi, self.range_bins_m, vmin=0, vmax=0.1)    #0.001)
        self.ax_1.set_title("Radar 1 polar beamformed map")


        # --- RADAR 2 POLAR PLOT
        self.fig_2  = plt.figure(figsize=(6, 6))
        self.ax_2   = self.fig_2.add_subplot(111, projection="polar")
        self.im_2   = configure_ax_bf( self.ax_2, self.phi, self.range_bins_m, vmin=0, vmax=0.1)    #0.001)
        self.ax_2.set_title("Radar 2 polar beamformed map")


        # --- FUSED CARTESIAN PLOT
        self.fig_cart = plt.figure(figsize=(7, 6))
        self.ax_cart = self.fig_cart.add_subplot(111)
        empty_cart = np.zeros_like(self.X, dtype=np.float32)

        self.im_cart = self.ax_cart.imshow(empty_cart.T, extent=self.cart_extent, origin="lower", aspect="equal", cmap="jet", vmin=0.0, vmax=1.0 ) # vmax=0.001

        # Radar positions displayed as (y, x) -- NOTE : this is done so as to have the Y-axis horizontal
        self.ax_cart.scatter(self.radar1_pose[1], self.radar1_pose[0], marker="^", s=100, label="Radar 1")
        self.ax_cart.scatter(self.radar2_pose[1], self.radar2_pose[0], marker="^", s=100, label="Radar 2")

        self.ax_cart.set_xlabel("y [m]  (left/right)")
        self.ax_cart.set_ylabel("x [m]  (forward)")
        self.ax_cart.set_title("Fused Cartesian radar view")
        self.ax_cart.grid(True)
        self.ax_cart.legend()

        self.ax_cart.invert_xaxis()


        #   INITIALIZE DETECTION WINDOW
        #   ---------------------------  shows "Jerry Detected" / "No Jerry" and the current radar detection rates / Cartesian EMA values
        #
        self.det_fig, self.det_ax = plt.subplots(figsize=(5, 2.5))
        self.det_text = self.det_ax.text(0.5, 0.5, "Starting...", ha="center", va="center", fontsize=16)
        self.det_ax.set_xticks([])
        self.det_ax.set_yticks([])



        #   Connect GUI close events to the stop path
        #   -----------------------------------------       NOTE : CLOSING ANY WINDOW STOPS THE WHOLE PIPELINE
        for fig in [self.fig_1, self.fig_2, self.fig_cart, self.det_fig]:
            fig.canvas.mpl_connect("close_event", self.on_close)

        self.accept("escape", self.request_shutdown)        # window close and Escape both call 'request_shutdown()'


        #   Register the realtime update loop
        #   ---------------------------------
        self.taskMgr.add(self.updateTask, "updateTask")     #  'updateTask()' called repeatedly by Panda3D :
                                                            #   Purpose :   it reads frames from q1/q2, synchronizes radar pairs, 
                                                            #               updates plots, runs detection, saves data, writes logs, and appends MP4 frames.
                                                            #   NOTE : does not process a frame immediately, rather tells panda to call 'updateTask()' repeatedly while the app is running




    def _make_detector_bins(self):
        """Build the range-bin ROI used by the per-radar Jerry detectors.
        
        The method reads ``detector_range_start_bin`` and ``detector_range_stop_bin``
        from ``cfg_radar``. If they are not provided, it uses a safe default window
        near bins 15 to 40, clipped to the available number of range bins.
        
        Returns:
            numpy.ndarray: One-dimensional array of valid integer range-bin indices.
        """
        n_range_bins = len(self.range_bins_m)
        default_start = min(15, max(0, n_range_bins - 1))
        default_stop = min(40, n_range_bins)
        start = int(self.cfg_radar.get("detector_range_start_bin", default_start))
        stop = int(self.cfg_radar.get("detector_range_stop_bin", default_stop))
        start = max(0, min(start, n_range_bins - 1))
        stop = max(start + 1, min(stop, n_range_bins))
        return np.arange(start, stop)
    


    def request_shutdown(self, event=None):
        """
        Sets the shared 'multiprocessing.Event', close recorders in the consumer,
        close Matplotlib windows, then ask Panda3D's loop to exit via userExit().
        """
        if self.closed:
            return

        print("-- Plot window closed. Stopping streaming...")
        self.closed = True
        self.is_closing = True
        self.stop_event.set()
        self.close_recorders()

        try:
            plt.close("all")
        except Exception:
            pass

        try:
            self.userExit()
        except Exception:
            pass


    def on_close(self, event):
        """Handle a Matplotlib window close event.
        
        Any figure window close is treated as a request to stop the complete streaming
        pipeline. The actual cleanup is delegated to ``request_shutdown`` so that all
        shutdown paths share the same logic.
        
        Args:
            event: Matplotlib close-event object. It is not used directly.
        """
        self.request_shutdown()


    def close_recorders(self):
        """Close file-based recorders owned by the consumer process.
        
        This closes the ImageIO/FFmpeg MP4 writer when recording is enabled.
        Closing the writer is required to finalize the MP4 container correctly. The
        method is idempotent: after closing, ``self.video_writer`` is set to ``None``.
        """
        if self.video_writer is not None:
            try:
                self.video_writer.close()
                print(f"-- Saved Cartesian MP4 to: {self.video_path}")
            except Exception as exc:
                print("-- Video writer close error:", exc)
            self.video_writer = None


    def save_dataset_metadata(self):
        """Write static dataset metadata and coordinate axes to disk.
        
        The metadata file documents the recording name, experiment folder, Cartesian
        grid limits, radar poses, range/azimuth bins, saved tensor layout, producer
        logic, Doppler mode, and CFAR/config values. The x and y axes are also saved as
        ``x_axis.npy`` and ``y_axis.npy``.
        
        Side effects:
            Creates/overwrites ``metadata.json``, ``x_axis.npy``, and ``y_axis.npy`` in
            ``self.dataset_dir``.
        """
    
        metadata = {
            "description"           : "Two-radar Cartesian beamformed dataset with realtime processing refactored from Windows Implementation",
            "recording_name"        : self.recording_name,
            "experiment_dir"        : str(self.experiment_dir),
            "file_format"           : "one .npz file per synchronized radar frame pair",
            "cnn_input_shape"       : "(2, H, W)",
            "channel_0"             : "Radar 1 Cartesian beamformed map",
            "channel_1"             : "Radar 2 Cartesian beamformed map",

            "axis_order_saved"      : {
                                        "cart_1": "(H, W) = (y, x)",
                                        "cart_2": "(H, W) = (y, x)",
                                        "cnn_input": "(C, H, W) = (radar_channel, y, x)",
                                      },

            "coordinate_convention" : {
                                        "x": "forward [m]",
                                        "y": "left/right [m]",
                                        "azimuth_deg": "0 deg means forward",
                                      },

            "radar1_pose"           : list(self.radar1_pose),
            "radar2_pose"           : list(self.radar2_pose),
            "cart_x_min_m"          : float(self.cfg_radar["cart_x_min_m"]),
            "cart_x_max_m"          : float(self.cfg_radar["cart_x_max_m"]),
            "cart_y_min_m"          : float(self.cfg_radar["cart_y_min_m"]),
            "cart_y_max_m"          : float(self.cfg_radar["cart_y_max_m"]),
            "cart_res_m"            : float(self.cfg_radar["cart_res_m"]),
            "x_axis_file"           : "x_axis.npy",
            "y_axis_file"           : "y_axis.npy",
            "range_bins_m"          : self.range_bins_m.astype(float).tolist(),
            "azimuth_deg"           : self.azimuth_deg.astype(float).tolist(),
            "producer_logic"        : "dense realtime beamforming refactored from Windows Implementation",
            "doppler_mode"          : bool(self.cfg_radar.get("doppler", False)),
            "cfar"                  : {k: _as_jsonable(v) for k, v in self.cfg_cfar.items()},
        }

        with open(self.dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        np.save(self.dataset_dir / "x_axis.npy", self.x_axis)
        np.save(self.dataset_dir / "y_axis.npy", self.y_axis)



    def save_frame_pair(self, cart_1, cart_2, fused_cart, detection_summary):
        """Save one synchronized dual-radar Cartesian frame pair for later ML use.
        
        The method stores radar 1 and radar 2 Cartesian maps as a two-channel tensor
        ``cnn_input`` with shape ``(2, H, W)``. It also saves the individual Cartesian
        maps, fused map, frame index, timestamp, and detection summary, then appends an
        entry to ``index.jsonl``.
        
        Args:
            cart_1: Radar-1 Cartesian power map.
            cart_2: Radar-2 Cartesian power map.
            fused_cart: Fused Cartesian map for this synchronized frame pair.
            detection_summary: Dictionary containing detection/debug metrics.
        
        Side effects:
            Writes a ``frame_XXXXXX.npz`` file and appends one JSON line to the dataset
            index when dataset recording is enabled and the frame interval matches.
        """
        if not self.record_dataset:
            return
        if self.frame_pair_idx % self.record_data_frame_every_n_pairs != 0:
            return

        cart_1 = cart_1.astype(np.float32)
        cart_2 = cart_2.astype(np.float32)
        cnn_input = np.stack([cart_1, cart_2], axis=0).astype(np.float32)
        timestamp = time.time()

        filename = f"frame_{self.frame_pair_idx:06d}.npz"

        
        # NOTE : save multiple np arrays into one container file (i.e. 'frame.npz') that stores the uncompressed arrays
        np.savez(                                       # NOTE : 'np.savez()' opens, writes, and closes that file by itself
            self.frames_dir / filename,
            cnn_input=cnn_input,                        # NOTE : np.savez is faster than np.savez_compressed for live recording.
            cart_1=cart_1,
            cart_2=cart_2,
            fused_cart=fused_cart.astype(np.float32),
            frame_idx=self.frame_pair_idx,
            timestamp=timestamp,
            detection_json=json.dumps(detection_summary),
        )

        index_entry = {
            "frame_idx"             :   self.frame_pair_idx,
            "timestamp"             :   timestamp,
            "file"                  :   f"frames/{filename}",
            "shape"                 :   list(cnn_input.shape),
            "detected"              :   bool(detection_summary["detected"]),
            "radar_detection_rates" :   detection_summary["radar_detection_rates"],
            "cartesian_ema"         :   detection_summary["cartesian_ema"],
        }
        
        with open(self.index_path, "a") as f:                   # NOTE : 'with open()' block closes the file automatically after each frame
            f.write(json.dumps(index_entry) + "\n")




    def save_cartesian_plot_frame_to_mp4(self):
        """Append the current Cartesian Matplotlib figure to the MP4 recording.
        
        The method captures the rendered RGBA canvas, converts it to RGB, and sends it
        to the ImageIO video writer. It respects ``record_mp4`` and
        ``record_video_every_n_pairs``.
        
        Side effects:
            Appends one video frame to ``cartesian.mp4`` when recording is enabled.
        """
        if not self.record_mp4 or self.video_writer is None:
            return
        if self.frame_pair_idx % self.record_video_every_n_pairs != 0:
            return
        try:
            self.fig_cart.canvas.draw()
            frame_rgba = np.asarray(self.fig_cart.canvas.buffer_rgba())
            frame_rgb = np.ascontiguousarray(frame_rgba[:, :, :3].copy())
            self.video_writer.append_data(frame_rgb)
        except Exception as exc:
            print("-- MP4 frame write error:", exc)

    def update_logs(self, summary):
        """Append per-frame signal and detection information to JSONL logs.
        
        Every processed synchronized frame writes a signal event to
        ``signal_log.jsonl``. Frames with a positive final detection are also written to
        ``jerry_log.jsonl``.
        
        Args:
            summary: Detection summary dictionary produced by ``updateTask``.
        
        Side effects:
            Appends JSON lines to the experiment log files.
        """
        signal_event = {
            "time"                  : datetime.now().isoformat(),
            "frame_idx"             : self.frame_pair_idx,
            "radar_detection_rates" : summary["radar_detection_rates"],
            "radar_active_bins"     : summary["radar_active_bins"],
            "cartesian_ema"         : summary["cartesian_ema"],
            "cartesian_roi_max"     : summary["cartesian_roi_max"],
            "detected"              : summary["detected"],
        }
        with open(self.signal_log_path, "a") as f:
            f.write(json.dumps(signal_event) + "\n")

        if summary["detected"]:
            with open(self.detection_log_path, "a") as f:
                f.write(json.dumps(signal_event) + "\n")

    def update_detection_display(self, summary):
        """Update the small textual detection-status figure.
        
        The display turns red and says "Jerry Detected" when the final detection flag is
        true. Otherwise it turns green and says "No Jerry". The label also includes the
        recent per-radar detection rates and Cartesian EMA for debugging.
        
        Args:
            summary: Detection summary dictionary produced by ``updateTask``.
        """
        if summary["detected"]:
            self.det_ax.set_facecolor("red")
            label = (
                "Jerry Detected\n"
                f"R1 {summary['radar_detection_rates'][0]:.0%} | "
                f"R2 {summary['radar_detection_rates'][1]:.0%} | "
                f"Cart EMA {summary['cartesian_ema']:.2f}"
            )
        else:
            self.det_ax.set_facecolor("green")
            label = (
                "No Jerry\n"
                f"R1 {summary['radar_detection_rates'][0]:.0%} | "
                f"R2 {summary['radar_detection_rates'][1]:.0%} | "
                f"Cart EMA {summary['cartesian_ema']:.2f}"
            )
        self.det_text.set_text(label)
        self.det_fig.canvas.draw_idle()

    def update_peak_marker(self, peak):
        """Move the Cartesian peak marker to the latest detected/localized point.
        
        The previous marker is removed before drawing a new one. If ``peak`` is
        ``None``, no marker is shown.
        
        Args:
            peak: ``None`` or a dictionary containing ``x_m`` and ``y_m`` coordinates
                in meters.
        """
        if self.last_peak_artist is not None:
            try:
                self.last_peak_artist.remove()
            except Exception:
                pass
            self.last_peak_artist = None
        if peak is not None:
            self.last_peak_artist = self.ax_cart.scatter(
                peak["y_m"],
                peak["x_m"],
                marker="x",
                s=140,
                linewidths=3,
                label="Detection peak",
            )

    def updateTask(self, task):
        """Process queued radar frames, update visualizations, detect, log, and record.
        
        This is the Panda3D task-loop callback executed repeatedly inside the consumer
        process. It drains the latest frame from each radar queue, waits until one frame
        from both radars is available, updates polar plots, projects each radar map to
        the global Cartesian grid, fuses the Cartesian maps, runs per-radar and
        Cartesian detectors, updates displays/logs, optionally records dataset/MP4
        outputs, and advances the synchronized frame-pair counter.
        
        Args:
            task: Panda3D task object supplied by the task manager.
        
        Returns:
            direct.task.Task.done: When the shared stop event is set and shutdown has
                been requested.
            direct.task.Task.cont: While streaming should continue.
        """
        if self.stop_event.is_set():
            self.request_shutdown()
            return Task.done

        try:
            for pid, q in enumerate((self.q1, self.q2)):
                while not q.empty():
                    msg = q.get_nowait()
                    if msg[0] == "bev":
                        self.latest_msg[pid] = msg[1]
                        self.msg_count.add(pid)
        except Exception as exc:
            print("-- Consumer queue error:", exc)


        if self.msg_count == {0, 1}:    # Producer sends: ("bev", bf_output)   =>  self.latest_msg[pid] == bf_output

            bf_1 = np.asarray(self.latest_msg[0], dtype=np.float32)
            bf_2 = np.asarray(self.latest_msg[1], dtype=np.float32)


            # Polar maps for visualization
            # ----------------------------
            bf_polar_1 = normalize_for_display(bf_1, exponent=self.cfg_radar.get("display_exponent", 1))
            bf_polar_2 = normalize_for_display(bf_2, exponent=self.cfg_radar.get("display_exponent", 1))

            self.im_1.set_array(bf_polar_1.ravel())
            self.im_2.set_array(bf_polar_2.ravel())
            self.fig_1.canvas.draw_idle()
            self.fig_2.canvas.draw_idle()



            # Raw-ish beamformed radar acquisitions for data-gathering
            # --------------------------------------------------------
            bf_raw_1 = np.abs(bf_1).astype(np.float32)
            bf_raw_2 = np.abs(bf_2).astype(np.float32)


            # Cartesian maps for DISPLAY
            # ---------------------------------------------
            cart_1_display = radar_power_on_cartesian_grid(
                power=bf_polar_1,
                azimuth_bins_deg=self.azimuth_deg,
                range_bins_m=self.range_bins_m,
                radar_pose=self.radar1_pose,
                x_grid=self.X,
                y_grid=self.Y,
            )
            cart_2_display = radar_power_on_cartesian_grid(
                power=bf_polar_2,
                azimuth_bins_deg=self.azimuth_deg,
                range_bins_m=self.range_bins_m,
                radar_pose=self.radar2_pose,
                x_grid=self.X,
                y_grid=self.Y,
            )


            # Cartesian maps for DATA RECORDING (in order to train CNN)
            # ---------------------------------------------------------

            cart_1_raw = radar_power_on_cartesian_grid(
                power=bf_raw_1,
                azimuth_bins_deg=self.azimuth_deg,
                range_bins_m=self.range_bins_m,
                radar_pose=self.radar1_pose,
                x_grid=self.X,
                y_grid=self.Y,
            )
            cart_2_raw = radar_power_on_cartesian_grid(
                power=bf_raw_2,
                azimuth_bins_deg=self.azimuth_deg,
                range_bins_m=self.range_bins_m,
                radar_pose=self.radar2_pose,
                x_grid=self.X,
                y_grid=self.Y,
            )



            # Fused Cartesian map for visualization only
            # ------------------------------------------

            fused_cart_raw = cart_1_raw + cart_2_raw                #   for dataset/debug and optional Cartesian peak localization
            fused_cart_display = cart_1_display + cart_2_display    #   normalized only for the heatmap


            #  NOTE    : normalize the fused Cartesian map (rescales fused_cart so its maximum value becomes 1.0)
            #  PURPOSE : VISUALISATION  (display heatmaps)
            #            -- every frame is comparable since normalized 
            #               => easier to interpret
            max_val = float(np.max(fused_cart_display)) if fused_cart_display.size else 0.0
            if max_val > 0:
                fused_cart_display = fused_cart_display / max_val



            #   RUN DETECTION ON RADAR 1 and 2 (separately)
            #   -------------------------------------------

            radar1_1D_range_profile = np.abs(bf_1).max(axis=0)      # NOTE : collapse the 2D beamformed map into a 1D range profile by taking the strongest value across all angles for every range bin.
            radar2_1D_range_profile = np.abs(bf_2).max(axis=0)

            #   Detect motion for each Radar (in the ROI)
            r1_detected, r1_rate, r1_bins, r1_noise, r1_thr, r1_roi_max = self.detectors[0].update_detection(radar1_1D_range_profile)
            r2_detected, r2_rate, r2_bins, r2_noise, r2_thr, r2_roi_max = self.detectors[1].update_detection(radar2_1D_range_profile)

            #   Detection Criterion : target detected whenever one of the radars detects motion     -- idea : system says "Jerry detected" : IF one of the 2 detectors (radar 1/2) detects activity.
            detected = bool(r1_detected or r2_detected)      



            #   CARTESIAN DETECTION VISUALISATION
            #   ---------------------------------     NOTE : for diagnostic/localization ONLY -- DOES NOT create a Jerry detection by itself
            #    
            cart_detected, cart_ema, cart_roi_max, peak = self.cart_detector.update(fused_cart_raw, self.X, self.Y)

            display_peak = peak if detected else None       # show peak marker only if the radar 1/2 detector actually fires


            #   structured packet containing all detection information for display/logging
            summary = {
                "detected"              : detected,
                "radar_detected"        : [bool(r1_detected), bool(r2_detected)],
                "radar_detection_rates" : [float(r1_rate), float(r2_rate)],
                "radar_active_bins"     : [int(r1_bins), int(r2_bins)],
                "radar_noise_floor"     : [float(r1_noise), float(r2_noise)],
                "radar_threshold"       : [float(r1_thr), float(r2_thr)],
                "radar_roi_max"         : [float(r1_roi_max), float(r2_roi_max)],

                # for debugging : NOT USED in FINAL DETECTION
                "cartesian_detected"            : bool(cart_detected),
                "cartesian_used_for_detection"  : False,
                "cartesian_ema"                 : float(cart_ema),
                "cartesian_roi_max"             : float(cart_roi_max),
                "peak"                          : display_peak,
            }


            # UPDATE the LIVE FUSED HEATMAPS
            # ------------------------------
            self.im_cart.set_array(fused_cart_display.T)        # NOTE : visualisation uses normalized frame

            self.update_peak_marker(display_peak)               # NOTE : display peak on fused cartesian plot

            self.ax_cart.set_title("Fused Cartesian radar view" + (" -- Jerry detected" if detected else ""))
            self.fig_cart.canvas.draw_idle()


            # update textual detection panel and write detection info to the logs
            # -------------------------------------------------------------------
            self.update_detection_display(summary)
            self.update_logs(summary)

            # Save one synchronized radar-frame pair (HxWxC=2) as new "frame-file" into experiment folder
            self.save_frame_pair(cart_1_raw, cart_2_raw, fused_cart_raw, summary)

            # Record Cartesian plot to MP4
            self.save_cartesian_plot_frame_to_mp4()

            # Count this synchronized frame pair
            self.frame_pair_idx += 1

            # Wait for a fresh frame from both radars next time
            self.msg_count.clear()

            QtWidgets.QApplication.processEvents()
            plt.pause(0.001)

        return Task.cont


def consumer(q1, q2, cfg_radar, stop_event, cfg_cfar=None):
    """
    Consumer process.

    Receives beamformed polar maps from two radar producers, displays them,
    projects them to Cartesian coordinates, records CNN-ready frame pairs,
    records the Cartesian plot as an MP4, and runs the WINDOWS IMPLEMENTATION detector.

    Stop handling is intentionally kept from project_3 because this was the
    Linux-tested path: the consumer owns the MP4 writer and therefore installs
    its own SIGINT/SIGTERM handler to close files before the process exits.
    """

    app = None
    cfg_cfar = cfg_cfar or {}

    # ----------------------------------------------------------------------
    def handle_stop_signal(signum, frame):
        """
        Called when the consumer process receives a stop signal.

        Handle Ctrl-C / termination inside the consumer process too.

        Purpose/Context : this is important because the MP4 writer exists in
        this process, not in the main process.
        """
        print("-- Consumer received stop signal. Closing MP4 writer...")
        stop_event.set()

        if app is not None:
            app.close_recorders()

            try:
                plt.close("all")
            except Exception:
                pass

            try:
                app.userExit()
            except Exception:
                pass
    # ----------------------------------------------------------------------

    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    try:
        app = MyApp(q1, q2, cfg_radar, cfg_cfar, stop_event)
        app.run()

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        if app is not None:
            app.close_recorders()

            try:
                plt.close("all")
            except Exception:
                pass

            try:
                app.destroy()
            except Exception:
                pass


def main(cfg_radar, cfg_cfar):
    """
    Start two radar producers and one consumer.
    """

    stop_event = Event()

    q_main_1 = Queue(maxsize=1)
    q_main_2 = Queue(maxsize=1)

    exp_path = cfg_radar["raw_dir"]

    exp_name_r1 = "radar1"
    exp_name_r2 = "radar2"

    producers = [
        Process(
            target=producer_real_time_1843_SAVE_DOPPLER,
            args=(q_main_1, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.180", stop_event, exp_name_r1, exp_path),
            daemon=True,
        ),
        Process(
            target=producer_real_time_1843_SAVE_DOPPLER,
            args=(q_main_2, cfg_radar, cfg_cfar, 4099, 5000, "192.168.33.32", "192.168.33.182", stop_event, exp_name_r2, exp_path),
            daemon=True,
        ),
    ]

    consumers = [
        Process(target=consumer, args=(q_main_1, q_main_2, cfg_radar, stop_event, cfg_cfar), daemon=False)
    ]

    for p in producers:
        p.start()
    for c in consumers:
        c.start()

    print("-- Streaming started.")

    try:
        while not stop_event.is_set():
            if any(not c.is_alive() for c in consumers):
                stop_event.set()
                break

            time.sleep(0.2)

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        print("-- Shutting down...")

        stop_event.set()

        # Stop producers first so no more frames are pushed.
        for p in producers:
            p.join(timeout=2.0)

        for p in producers:
            if p.is_alive():
                p.terminate()

        for p in producers:
            p.join()

        # Give the consumer enough time to close the imageio/FFmpeg writer.
        # This is what finalizes the MP4 file correctly.
        for c in consumers:
            c.join(timeout=10.0)

        for c in consumers:
            if c.is_alive():
                print("-- Consumer did not exit cleanly; terminating it.")
                c.terminate()

        for c in consumers:
            c.join()

        print(f"-- Experiment saved to: {cfg_radar['experiment_dir']}")
        print("-- Shutdown complete.")
