# COM-304 Radars Rat-Detection Project (Linux)


![alt text](<Screenshot from 2026-05-29 12-16-46.png>)


This repository contains the code for the COM-304 project for Linux-oriented real-time rat detection pipeline using Texas Instruments mmWave radar hardware and DCA1000EVM capture boards. The codebase reads live ADC data, performs range/Doppler processing and azimuth beamforming, projects radar power maps into Cartesian coordinates, fuses one or two radar streams, visualizes detections, and saves experiment data for later analysis or machine-learning workflows.


### Features

- Live capture from TI AWR1843BOOST-style mmWave radar hardware through DCA1000EVM.
- Linux-compatible serial ports such as `/dev/ttyACM0` and `/dev/ttyACM1`.
- Single-radar and dual-radar streaming pipelines.
- Real-time beamforming into polar radar maps.
- Cartesian projection of radar power maps.
- Two-radar Cartesian fusion.
- Range-bin based activity detection for “Jerry” / rat detection.
- Optional CFAR and Doppler processing modes.
- Experiment recording with raw `.bin` streams, `.npz` Cartesian frame tensors, metadata, logs, and MP4 visualizations.


## Info 


### Repository Structure

```text
COM-304_Rat-Detection_LINUX/
├── environment.yml                  # Full exported Conda environment
├── README.md
├── configs/                         # TI mmWave .cfg radar profiles
│   ├── D_doppler_config.cfg
│   └── my_config.cfg
└── src/
    ├── configuration/               # Radar configuration helpers
    ├── mmwave/                      # mmWave data loading utilities
    ├── mmwavecapture/               # Radar and DCA1000 control code
    ├── processing/                  # FFT, CFAR, DBSCAN, beamforming utilities
    ├── streaming/                   # Live capture, streaming, fusion, and detection scripts
    ├── utils/                       # Coordinate transforms and radar utility functions
    └── visualization/               # Plotting helpers
```


### Main processing components

- `src/streaming/prod_dca.py`: live DCA1000 producer, range FFT, optional background subtraction, Doppler filtering, beamforming, and queue output.
- `src/streaming/realtime_streaming_1r.py`: single-radar real-time visualization and recording loop.
- `src/streaming/realtime_streaming_2r_fused.py`: dual-radar synchronization, Cartesian fusion, logging, recording, and Jerry detection.
- `src/processing/processing.py`: CFAR, beamforming, DBSCAN, FFT-related utilities.
- `src/utils/utils.py`: polar-to-Cartesian projection, normalization, antenna geometry utilities.
- `src/mmwavecapture/`: radar and DCA1000 control helpers.



### Hardware Requirements

This project is designed around TI mmWave radar capture hardware:

- TI AWR1843BOOST radar board
- DCA1000EVM capture board
- Linux host machine
- USB connections for radar configuration/data serial ports
- Ethernet connection for DCA1000 UDP capture

For dual-radar experiments, the code expects two radar/DCA1000 streams with separate host/DCA IP and UDP port settings.

### Software Requirements

The `environment.yml` file defines a Conda environment named `radar` with Python 3.13 and the main runtime dependencies.



## Running the Project


### 1. Single Radar Setup
### 1.1 Start a single radar manually


```bash
sudo systemctl stop ModemManager
```

```bash
ip -br link
```

```bash
sudo ip addr flush dev <RADAR1_IFACE>
sudo ip addr add 192.168.33.30/24 dev <RADAR1_IFACE>
sudo ip link set <RADAR1_IFACE> up

cd .../304_Rat-Detection_LINUX/src
conda activate radar
python -m streaming.start_radar_1
```

This initializes the DCA1000EVM, configures the radar, starts recording, and starts the radar sensor.


### 1.2 Run single-radar streaming


In a second terminal:

```bash

cd .../304_Rat-Detection_LINUX/src
conda activate radar
python -m streaming.stream_1r
```

Useful options:

```bash
--config my_config       # Radar .cfg file name without .cfg
--cfar                   # Enable CFAR processing
--doppler                # Enable Doppler mode
--save_raw_dt            # Save raw ADC data
--exp_name test          # Base experiment name
--beam-width 180         # Beamforming angular width in degrees
--beam-center 90         # 90 degrees means straight ahead
```


---


### 2. Dual-radar real-time setup

The dual-radar setup is more delicate because both DCA1000 boards need to stream on the network without conflicting.

The high-level process is:

1. Configure Radar 2 so it uses a different DCA1000 setup.
2. Start Radar 1.
3. Detach Radar 1 USB cables after it is streaming.
4. Switch Radar 2 DCA1000 jumper to use the configured IP mode.
5. Start Radar 2.
6. Run the dual-radar viewer.

---

### 2.1 Radar 2 pre-configuration

Before the dual-radar run, configure the second radar.

#### Important jumper setting

For Radar 2 configuration, the DCA1000EVM `SW2.6` jumper must be on:

```text
pin 11 = GND
```

#### Configure Ethernet temporarily

First, find the ethernet interface name associated with the radar using :

```bash
ip -br link
```


Use the second radar Ethernet interface, but temporarily assign the default host IP:

```bash
sudo ip addr flush dev <RADAR2_IFACE>
sudo ip addr add 192.168.33.30/24 dev <RADAR2_IFACE>
sudo ip link set <RADAR2_IFACE> up
```

Verify:

```bash
ip -4 addr show dev <RADAR2_IFACE>
ip route get 192.168.33.180
```

You want the route to use `<RADAR2_IFACE>` with source `192.168.33.30`.

#### Run the Radar 2 configuration script

```bash
cd .../304_Rat-Detection_LINUX/src
conda activate radar
python -m configuration.configure_radar_2
```

After this succeeds, shut down Radar 2.

Then delete the temporary route:

```bash
sudo route delete -net 192.168.33.0/24
```

---

### 2.2 Start Radar 1

#### Radar 1 jumper setting

For Radar 1, the DCA1000EVM `SW2.6` jumper should be on:

```text
pin 11 = GND
```

#### Configure Radar 1 Ethernet

```bash
sudo ip addr flush dev <RADAR1_IFACE>
sudo ip addr add 192.168.33.30/24 dev <RADAR1_IFACE>
sudo ip link set <RADAR1_IFACE> up
```

Verify:

```bash
ip -4 addr show dev <RADAR1_IFACE>
ip route get 192.168.33.180
```

#### Start Radar 1

Terminal 1:

```bash
cd .../304_Rat-Detection_LINUX/src
conda activate radar
python -m streaming.start_radar_1
```

Once Radar 1 is streaming, detach the two USB cables from Radar 1.

Keep these connected:

- Radar 1 power cable
- Radar 1 Ethernet cable

Then delete the route:

```bash
sudo route delete -net 192.168.33.0/24
```

If something does not work, for example if you forgot to plug in the USB cables:

1. Close the terminal.
2. Fix the cable issue.
3. Power off/on the radar.
4. Restart the procedure.

---

### 2.3 Prepare Radar 2

Switch Radar 2 DCA1000EVM `SW2.6` jumper from:

```text
pin 11 = GND
```

to:

```text
pin 6 = USER_SW1
```

This tells the DCA1000EVM to use the newly configured IP mode from the Radar 2 configuration script.

Now connect the second AWR1843BOOST board via USB.

---

### 2.4 Configure Radar 2 Ethernet

Assign the second host IP to the Radar 2 Ethernet interface:

```bash
sudo ip addr flush dev <RADAR2_IFACE>
sudo ip addr add 192.168.33.32/24 dev <RADAR2_IFACE>
sudo ip link set <RADAR2_IFACE> up
```

---

### 2.5 Start Radar 2

Terminal 2:

```bash
cd .../304_Rat-Detection_LINUX/src
conda activate radar
python -m streaming.start_radar_2
```

Run these commands :

```bash
sudo ip addr flush dev <enx_RADAR_1_INTERFACE>
sudo ip addr flush dev <enx_RADAR_2_INTERFACE>

sudo ip addr add 192.168.33.30/24 dev <enx_RADAR_1_INTERFACE>
sudo ip addr add 192.168.33.32/24 dev <enx_RADAR_2_INTERFACE>

sudo ip link set <enx_RADAR_1_INTERFACE> up
sudo ip link set <enx_RADAR_2_INTERFACE> up
```


### 2.6 Temporarily Increase Linux UDP Reciever buffer size 

```bash
sudo sysctl -w net.core.rmem_max=268435456
sudo sysctl -w net.core.rmem_default=268435456
```

---

### 2.6 Start the dual-radar viewer

Open another terminal:

```bash
cd .../304_Rat-Detection_LINUX/src
conda activate radar
python -m streaming.stream_2r
```

You should see radar data from both radars, usually in four Matplotlib windows.


#### Example Run :

From the `src` directory:

```bash
python -m streaming.stream_2r \
  --config my_config \
  --exp_name test \
  --mid_gap 0.2 \
  --beam-width 90 \
  --beam-center 90 \
  --max-range-m 1.5
```

Useful options:

```bash
--config my_config              # Radar .cfg file name without .cfg
--cfar                          # Enable CFAR
--doppler                       # Enable Doppler mode
--save_raw_dt                   # Save raw radar data
--exp_name test                 # Base experiment name
--mid_gap 0.2                   # Distance between the two radars in meters
--beam-width 90                 # Beamforming angular width in degrees
--beam-center 90                # 90 degrees means straight ahead
--max-range-m 1.5               # Optional short-range gate for rat detection
--no-bg-sub                     # Disable background subtraction in non-Doppler mode
--doppler-notch-bins 1          # Zero-velocity Doppler notch size
--doppler-snr-threshold 2.0     # Doppler SNR gate
--near_range_zero_bins 5        # Ignore very close noisy range bins
--far_range_zero_bins 0         # Optionally ignore far range bins
```

## Detection Logic

The dual-radar pipeline contains a range-bin activity detector named `JerryClassifier`. It collapses a beamformed radar map into a 1D range profile, estimates a noise floor outside the selected range-bin region, and tracks activity over a moving window. A rat/Jerry detection is reported when either radar detects sustained activity in the selected range region.

The fused Cartesian map is used for visualization and optional localization/debugging. The final detection decision is based on the per-radar range-bin detector.


## Output Data

Live runs create experiment folders under:

```text
Data_Live_Experiments/<experiment_name>_<timestamp>/
```

Typical output layout:

```text
Data_Live_Experiments/<recording>/
├── experiment_metadata.json
├── raw/
│   ├── radar1_raw.bin
│   └── radar2_raw.bin
├── cartesian_frames/
│   ├── metadata.json
│   ├── index.jsonl
│   ├── x_axis.npy
│   ├── y_axis.npy
│   └── frames/
│       ├── frame_000000.npz
│       ├── frame_000001.npz
│       └── ...
├── mp4/
│   └── cartesian.mp4
└── logs/
    ├── signal_log.jsonl
    └── jerry_log.jsonl
```

Each saved `.npz` frame can contain:

- `cnn_input`: model-ready tensor
- `cart_1`: Cartesian radar map from radar 1
- `cart_2`: Cartesian radar map from radar 2, for dual-radar mode
- `fused_cart`: fused Cartesian map, for dual-radar mode
- `frame_idx`: frame number
- `timestamp`: capture timestamp
- `detection_json`: detection/debug information


## Radar Configuration Files

Radar profiles are stored in `configs/`. The main scripts accept a config name without the `.cfg` suffix.

For example:

```bash
--config my_config
```

uses:

```text
configs/my_config.cfg
```

To inspect calculated radar parameters from a `.cfg` file, run from the `src` directory:

```bash
cd src
python -m streaming.configure my_config
```

This parses the TI mmWave config file and prints derived values such as:

- number of TX/RX antennas
- chirps per frame
- number of Doppler bins
- number of range bins
- range resolution
- max range
- Doppler resolution
- max Doppler velocity



## Troubleshooting

### `ModuleNotFoundError`

Make sure to run scripts from inside `src` with `python -m ...`.

### Serial permission denied

Add your user to the `dialout` group:

```bash
sudo usermod -a -G dialout $USER
newgrp dialout
```

### DCA1000 socket bind error

Make sure the host IP address in the code is assigned to your Ethernet interface and that no other process is using the same UDP ports.

### No radar data received

Check:

- radar board power and USB connection
- DCA1000 Ethernet connection
- serial ports `/dev/ttyACM0` and `/dev/ttyACM1`
- host/DCA IP addresses
- UDP config/data ports
- selected `.cfg` file
- firewall rules blocking UDP traffic

### MP4 file is empty or corrupted

Stop the process cleanly with `Ctrl+C` or by closing the plot window. The MP4 writer must be closed cleanly to finalize the file.

## Suggested Workflow

1. Confirm Linux sees the radar serial ports.
2. Configure the Ethernet interface for the DCA1000 board.
3. Activate the Conda environment.
4. Parse the `.cfg` file to verify radar parameters.
5. Start with single-radar capture.
6. Enable `--save_raw_dt` only when raw recording is needed.
7. Move to dual-radar fused streaming after both radar/DCA1000 streams are stable.
8. Inspect `Data_Live_Experiments/` outputs after each run.


## Usefull Links

https://github.com/mmwave-capture-std/mmwave-capture-std/

https://www.ti.com/tool/UNIFLASH

https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/

https://github.com/Janvdk5/COM-304-Rat-Detection

https://github.com/Tkemper2/COM-304-Radar-tuto

## License / Attribution

Portions of this project include third-party code located in src/mmwavecapture/
Copyright (c) 2023 Louie Lu louielu@cs.unc.edu
Licensed under the Clear BSD License.

The license terms are included in the headers of the relevant source files.

