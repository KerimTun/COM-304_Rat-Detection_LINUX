import os
import sys
import time
import json
import warnings
import signal
from pathlib import Path
from datetime import datetime

warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import numpy as np

from multiprocessing import Process, Queue, Event

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

# Prevent plot windows from jumping above everything else
matplotlib.rcParams["figure.raise_window"] = False

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-dark")


from PyQt5 import QtWidgets

import imageio.v2 as imageio

from .prod_dca import producer_real_time_1843_SAVE_DOPPLER

from visualization.visualization import configure_ax_bf
from utils.utils import radar_power_on_cartesian_grid, normalize_for_display




def consumer(q1, cfg_radar, stop_event):
    """
    Consumer process.

    Receives beamformed polar map from the radar producer, displays it,
    projects it to Cartesian coordinates, records CNN-ready frame pairs,
    and records the Cartesian plot as an MP4.

    """

    app = None

    # ----------------------------------------------------------------------
    def handle_stop_signal(signum, frame):
        """
        Called when the consumer process receives a stop signal.

        Handle Ctrl-C / termination inside the consumer process too.

        Purpose/Context : this is important because the MP4 writer exists in this process,
        not in the main process.
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

    signal.signal(signal.SIGINT, handle_stop_signal)        # NOTE : these lines (86-87) register the signal handler
    signal.signal(signal.SIGTERM, handle_stop_signal)       # -- INTUITION behind       : "When this child consumer process receives Ctrl-C or terminate, run handle_stop_signal()."
                                                            # -- PURPOSE / IMPORTANCE   : matters because the MP4 writer is not in the main process (but inside the consumer process). 
                                                            #                             => handling Ctrl-C only in main() is not enough. 

    try:
        app = MyApp(q1, cfg_radar, stop_event)  # NOTE : this starts the app (creates the GUI/plotting app)
        app.run()                                   #        and starts its event loop

    except KeyboardInterrupt:       # NOTE : catches Ctrl-C in Python form. 
        stop_event.set()            #        -- IDEA : sometimes, Ctrl-C appears as a Python KeyboardInterrupt exception rather than only as a low-level signal
                                    #                  -> this EXCEPT Block catches that case and sets the shared stop flag

    finally:                        # NOTE : Finally block runs whether the app exits normally, crashes, or receives Ctrl-C.
        if app is not None:
            app.close_recorders()   # NOTE : should always happen before the consumer process ends.

            try:
                plt.close("all")
            except Exception:
                pass

            try:
                app.destroy()       # NOTE : cleans up Panda3D resources
            except Exception:
                pass


class MyApp(ShowBase):
    """
    Real-time two-radar visualization and dataset recording app.
    """

    def __init__(self, queue_1, cfg_radar, stop_event):
        ShowBase.__init__(self)

        self.q1 = queue_1
        self.stop_event = stop_event

        self.latest_msg = None
        self.msg_count = set()

        self.cfg_radar = cfg_radar

        self.closed = False
        self.video_writer = None

        # Polar radar grid
        # -------------------------
        self.phi = cfg_radar["phi"]                         # Beamforming angles [rad]
        self.azimuth_deg = cfg_radar["azimuth_deg"]         # Physical azimuth [deg]
        self.range_bins_m = cfg_radar["range_bins_m"]       # Range bins [m]


        # Radar pose
        # -----------------------------------------
        self.radar1_pose = cfg_radar["radar1_pose"]


        # RECORDING SETUP
        # --------------------------
        self.record_dataset = True
        self.record_mp4 = True

        #  CREATE EXPERIMENT SPECIFIC DIRECTORY
        #  in which we store the :
        #     - cartesian plot MP4; 
        #     - Raw Data (.bin file) for each Radar; 
        #     - Beamformed Cartesian Frame Pairs (used as input for classifier)


        self.record_data_frame_every_n_pairs = 1
        self.record_video_every_n_pairs = 1        # change to 5 if to slow

        self.frame_pair_idx = 0
        self.video_fps = 10


        self.experiment_dir = Path(cfg_radar["experiment_dir"])
        self.recording_name = cfg_radar["recording_name"]

        self.dataset_dir = Path(cfg_radar["cartesian_dir"])
        self.frames_dir = self.dataset_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.dataset_dir / "index.jsonl"

        self.video_dir = Path(cfg_radar["mp4_dir"])
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.video_dir / "cartesian.mp4"



        if self.record_mp4:
            self.video_writer = imageio.get_writer(
                self.video_path,
                fps=self.video_fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )


        # Cartesian global fusion grid
        # ------------------------------------------------------
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

        # Display orientation:
        #   horizontal axis = y
        #   vertical axis   = x
        self.cart_extent = [
            self.y_axis.min(),
            self.y_axis.max(),
            self.x_axis.min(),
            self.x_axis.max(),
        ]

        self.save_dataset_metadata()



        # Plot section
        # ---------------------------------------------------------

        # --- RADAR 1 POLAR PLOT
        self.fig_1 = plt.figure(figsize=(6, 6))
        self.ax_1 = self.fig_1.add_subplot(111, projection="polar")
        self.im_1 = configure_ax_bf( self.ax_1, self.phi, self.range_bins_m, vmin=0, vmax=0.001)

        self.ax_1.set_title("Radar 1 polar beamformed map")


        # --- FUSED CARTESIAN PLOT
        self.fig_cart = plt.figure(figsize=(7, 6))
        self.ax_cart = self.fig_cart.add_subplot(111)

        empty_cart = np.zeros_like(self.X, dtype=np.float32)

        self.im_cart = self.ax_cart.imshow(
            empty_cart.T,
            extent=self.cart_extent,
            origin="lower",
            aspect="equal",
            cmap="jet",
            vmin=0.0,
            vmax=0.001,
        )

        # Radar positions displayed as (y, x) -- NOTE : this is done so as to have the Y-axis horizontal
        self.ax_cart.scatter(
            self.radar1_pose[1],
            self.radar1_pose[0],
            marker="^",
            s=100,
            label="Radar 1",
        )

        self.ax_cart.set_xlabel("y [m]  (left/right)")
        self.ax_cart.set_ylabel("x [m]  (forward)")
        self.ax_cart.set_title("Cartesian radar view")
        self.ax_cart.grid(True)
        self.ax_cart.legend()


        # NOTE : CLOSING ANY WINDOW STOPS THE WHOLE PIPELINE
        for fig in [self.fig_1, self.fig_cart]:
            fig.canvas.mpl_connect("close_event", self.on_close)

        self.taskMgr.add(self.updateTask, "updateTask")





    #              SHUT DOWN / CLEAN UP
    # ------------------------------------------------
    def on_close(self, event):
        """
        Called when any Matplotlib window is closed.
        """

        if self.closed:
            return

        print("-- Plot window closed. Stopping streaming...")
        self.closed = True
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

    def close_recorders(self):
        """
        Close the MP4 writer safely.
        """

        if self.video_writer is not None:
            try:
                self.video_writer.close()
                print(f"-- Saved Cartesian MP4 to: {self.video_path}")
                print(f"-- MP4 file has been safely closed!")
            except Exception as e:
                print("Video writer close error:", e)

            self.video_writer = None





    #                           DATASET RECORDING 
    # -------------------------------------------------------------------------

    def save_dataset_metadata(self):
        """
        Save metadata needed to interpret the recorded dataset.
        """

        metadata = {
            "description"           : "Single-radar Cartesian beamformed dataset",
            "recording_name"        : self.recording_name,
            "experiment_dir"        : str(self.experiment_dir),
            "file_format"           : "one .npz file radar frame",
            "cnn_input_shape"       : "(1, H, W)",
            "channel_0"             : "Radar 1 Cartesian beamformed map",
            "axis_order_saved"      : {
                                        "cart_1": "(H, W) = (y, x)",
                                        "cnn_input": "(C, H, W) = (radar_channel, y, x)",
                                      },
            "coordinate_convention" : {
                                        "x": "forward [m]",
                                        "y": "left/right [m]",
                                        "azimuth_deg": "0 deg means forward",
                                      },
            "radar1_pose"           : list(self.radar1_pose),
            "cart_x_min_m"          : float(self.cfg_radar["cart_x_min_m"]),
            "cart_x_max_m"          : float(self.cfg_radar["cart_x_max_m"]),
            "cart_y_min_m"          : float(self.cfg_radar["cart_y_min_m"]),
            "cart_y_max_m"          : float(self.cfg_radar["cart_y_max_m"]),
            "cart_res_m"            : float(self.cfg_radar["cart_res_m"]),
            "x_axis_file"           : "x_axis.npy",
            "y_axis_file"           : "y_axis.npy",
            "range_bins_m"          : self.range_bins_m.astype(float).tolist(),
            "azimuth_deg"           : self.azimuth_deg.astype(float).tolist(),
        }

        metadata_path = self.dataset_dir / "metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        np.save(self.dataset_dir / "x_axis.npy", self.x_axis)
        np.save(self.dataset_dir / "y_axis.npy", self.y_axis)



    def save_frame(self, cart_1):
        """
        Save one synchronized pair of Cartesian radar maps.

        Saved shapes:
            cart_1:    (H, W)
            cnn_input: (1, H, W)
        """

        if not self.record_dataset:
            return

        if self.frame_pair_idx % self.record_data_frame_every_n_pairs != 0:
            return

        cart_1 = cart_1.astype(np.float32)

        cnn_input = cart_1 #np.stack([cart_1, cart_2], axis=0).astype(np.float32)

        timestamp = time.time()

        filename = f"frame_{self.frame_pair_idx:06d}.npz"
        save_path = self.frames_dir / filename


        # NOTE : save multiple np arrays into one container file (i.e. 'frame.npz') that stores the uncompressed arrays
        np.savez(                                   # NOTE : 'np.savez()' opens, writes, and closes that file by itself
            save_path,                              
            cnn_input=cnn_input,                    # NOTE : np.savez is faster than np.savez_compressed for live recording.
            cart_1=cart_1,
            frame_idx=self.frame_pair_idx,
            timestamp=timestamp,
        )

        
        index_entry = {
            "frame_idx": self.frame_pair_idx,
            "timestamp": timestamp,
            "file": f"frames/{filename}",
            "shape": list(cnn_input.shape),
        }

        with open(self.index_path, "a") as f:           # NOTE : 'with open()' block closes the file automatically after each frame
            f.write(json.dumps(index_entry) + "\n")



    #                           MP4 recording
    # -------------------------------------------------------------------------

    def save_cartesian_plot_frame_to_mp4(self):
        """
        Save the current Cartesian Matplotlib figure as one MP4 frame.
        This records the plot with axes, labels, markers, and colors.
        """

        if not self.record_mp4:
            return

        if self.video_writer is None:
            return

        if self.frame_pair_idx % self.record_video_every_n_pairs != 0:
            return

        try:
            self.fig_cart.canvas.draw()

            frame_rgba = np.asarray(self.fig_cart.canvas.buffer_rgba())
            frame_rgb = frame_rgba[:, :, :3].copy()

            # Make frame memory-contiguous before passing it to imageio/FFmpeg
            frame_rgb = np.ascontiguousarray(frame_rgb)

            self.video_writer.append_data(frame_rgb)

        except Exception as e:
            print("MP4 frame write error:", e)



    #                           Main update loop
    # -------------------------------------------------------------------------

    def updateTask(self, task):
        """
        Update task.

        Reads both radar queues, updates plots, projects polar beamformed maps
        to Cartesian maps, saves one paired dataset sample, and writes one MP4
        frame.
        """

        if self.stop_event.is_set():
            self.close_recorders()

            try:
                plt.close("all")
            except Exception:
                pass

            try:
                self.userExit()
            except Exception:
                pass

            return Task.done

        try:
            while not self.q1.empty():
                msg = self.q1.get_nowait()

                if msg[0] == "bev":
                    self.latest_msg = msg[1]

        except Exception as e:
            print("Consumer queue error:", e)


        # Check if we have received a new messages from producer
        if self.latest_msg is not None:
            # Producer sends: ("bev", bf_output)    =>    self.latest_msg[pid] == bf_output

            bf_1 = self.latest_msg


            # Polar map for visualization
            # -------------------------------------------
            bf_polar_1 = normalize_for_display(bf_1)

            self.im_1.set_array(bf_polar_1.ravel())

            self.fig_1.canvas.draw_idle()


            # Raw-ish beamformed radar acquisitions for data-gathering
            # --------------------------------------------------------
            bf_raw_1 = np.abs(bf_1).astype(np.float32)



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

            # Save one synchronized radar-frame pair (HxWxC=2) as new "frame-file" into experiment folder
            self.save_frame(cart_1_raw)

            # ---------------------------------------------------------


            # Fused Cartesian map for visualization only
            # ------------------------------------------
            fused_cart = cart_1_display

            max_val = np.max(fused_cart)
            if max_val > 0:
                fused_cart = fused_cart / max_val

            self.im_cart.set_array(fused_cart.T)
            self.fig_cart.canvas.draw_idle()


            # Record Cartesian plot to MP4
            self.save_cartesian_plot_frame_to_mp4()

            # Count this synchronized frame pair
            self.frame_pair_idx += 1

            # Wait for a fresh frame from both radars next time
            self.msg_count.clear()

            QtWidgets.QApplication.processEvents()
            plt.pause(0.001)

        return Task.cont


def main(cfg_radar, cfg_cfar):
    """
    Start one radar producer and one consumer process.
    """

    stop_event  = Event()
    q_main_1    = Queue(maxsize=1)
    exp_path    = cfg_radar["raw_dir"]

    exp_name_r1 = "radar1"
    producers   = [
        Process(target=producer_real_time_1843_SAVE_DOPPLER, args=( q_main_1, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.180", stop_event, exp_name_r1, exp_path ), daemon=True, ),
    ]

    consumers = [ Process(target=consumer, args=(q_main_1, cfg_radar, stop_event), daemon=False) ]

    for p in producers: p.start()
    for c in consumers: c.start()

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
        for p in producers: p.join(timeout=2.0)

        for p in producers:
            if p.is_alive():
                p.terminate()

        for p in producers: p.join()

        # Give the consumer enough time to close the imageio/FFmpeg writer.
        # This is what finalizes the MP4 file correctly.
        for c in consumers: c.join(timeout=10.0)

        for c in consumers:
            if c.is_alive():
                print("-- Consumer did not exit cleanly; terminating it.")
                c.terminate()

        for c in consumers: c.join()

        print(f"-- Experiment saved to: {cfg_radar['experiment_dir']}")

        print("-- Shutdown complete.")
