import argparse
from pathlib import Path


C = 3e8  # speed of light [m/s]



def bit_count(x: int) -> int:
    """
    Count the number of enabled bits in an integer bitmask.

    In TI mmWave configuration files, RX and TX channels are often encoded
    as bitmasks. 
    
    For example:

        txEnable = 5

    In binary, 5 is:

        0b101

    meaning two TX antennas are enabled.

    Parameters
    ----------
    x : int
        Integer bitmask.

    Returns
    -------
    int
        Number of bits set to 1.
    """

    return bin(x).count("1")



def parse_cfg(path):
    """
    Parse a TI mmWave '.cfg' configuration file.

    This function reads the configuration file line by line and extracts the
    fields required for radar parameter calculation.

    It currently parses:

        - channelCfg
        - profileCfg
        - chirpCfg
        - frameCfg

    Other commands are ignored.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the '.cfg' file.

    Returns
    -------
    dict
        Dictionary containing parsed configuration data.

        Example structure:

        {
            "channelCfg": {...},
            "profileCfg": {
                profile_id: {...}
            },
            "chirpCfg": [
                {...},
                {...}
            ],
            "frameCfg": {...}
        }
    """

    cfg = {
        "channelCfg": None,
        "profileCfg": {},
        "chirpCfg": [],
        "frameCfg": None,
    }

    lines = Path(path).read_text().splitlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("%"):
            continue

        # Remove inline comments if any
        line = line.split("%")[0].strip()

        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd == "channelCfg":
            # channelCfg rxChannelEn txChannelEn cascading
            cfg["channelCfg"] = {
                "rxChannelEn": int(parts[1]),
                "txChannelEn": int(parts[2]),
                "cascading": int(parts[3]),
            }

        elif cmd == "profileCfg":
            # profileCfg profileId startFreq idleTime adcStartTime rampEndTime
            #            txOutPower txPhaseShifter freqSlopeConst txStartTime
            #            numAdcSamples digOutSampleRate hpf1 hpf2 rxGain
            profile_id = int(parts[1])

            cfg["profileCfg"][profile_id] = {
                "profileId": profile_id,
                "startFreq": float(parts[2]),          # GHz
                "idleTime": float(parts[3]),           # us
                "adcStartTime": float(parts[4]),       # us
                "rampEndTime": float(parts[5]),        # us
                "txOutPower": float(parts[6]),
                "txPhaseShifter": float(parts[7]),
                "freqSlopeConst": float(parts[8]),     # MHz/us
                "txStartTime": float(parts[9]),        # us
                "numAdcSamples": int(parts[10]),
                "digOutSampleRate": float(parts[11]),  # ksps
                "hpfCornerFreq1": float(parts[12]),
                "hpfCornerFreq2": float(parts[13]),
                "rxGain": float(parts[14]),
            }

        elif cmd == "chirpCfg":
            # chirpCfg startIdx endIdx profileId startFreqVar freqSlopeVar
            #          idleTimeVar adcStartTimeVar txEnable
            cfg["chirpCfg"].append({
                "startIdx": int(parts[1]),
                "endIdx": int(parts[2]),
                "profileId": int(parts[3]),
                "startFreqVar": float(parts[4]),
                "freqSlopeVar": float(parts[5]),
                "idleTimeVar": float(parts[6]),
                "adcStartTimeVar": float(parts[7]),
                "txEnable": int(parts[8]),
            })

        elif cmd == "frameCfg":
            # frameCfg chirpStartIdx chirpEndIdx numLoops numFrames
            #          framePeriodicity triggerSelect frameTriggerDelay
            cfg["frameCfg"] = {
                "chirpStartIdx": int(parts[1]),
                "chirpEndIdx": int(parts[2]),
                "numLoops": int(parts[3]),
                "numFrames": int(parts[4]),
                "framePeriodicity": float(parts[5]),   # ms
                "triggerSelect": int(parts[6]),
                "frameTriggerDelay": float(parts[7]),
            }

    return cfg


def chirps_used_in_frame(chirp_cfgs, frame_cfg):
    """
    Find which chirp configurations are used by the frame configuration.

    The 'frameCfg' command defines a chirp index range as : 
    
        chirpStartIdx -> chirpEndIdx

    This function checks which 'chirpCfg' entries overlap with that range.

    Parameters
    ----------
    chirp_cfgs : list of dict
        List of parsed chirp configurations.

    frame_cfg : dict
        Parsed frame configuration.

    Returns
    -------
    list of dict
        Chirp configurations used in the frame.
    """

    start = frame_cfg["chirpStartIdx"]
    end = frame_cfg["chirpEndIdx"]

    used = []

    for chirp in chirp_cfgs:
        chirp_start = chirp["startIdx"]
        chirp_end = chirp["endIdx"]

        # Check overlap with frame chirp range
        if chirp_end >= start and chirp_start <= end:
            used.append(chirp)

    return used


def compute_params(cfg, max_range_scale=0.9):
    """

    Extracts the required values from:
        - profileCfg
        - frameCfg
        - chirpCfg
        - channelCfg

    and computes Parameters w.r.t. the limitations of the TI ARW1843 Radar (limitations stated by the TI Document) :
        - chirps per frame
        - number of Doppler bins
        - number of range bins
        - range resolution
        - maximum unambiguous range
        - Doppler resolution
        - maximum Doppler velocity

    Parameters
    ----------

    cfg :   dict
        Dictionnary of key configuration fields (in config files)
    max_range_scale=0.9
        Scale factor of AWR1843 limitation (as specified in TI document)

        
    Returns
    -------
    dict : dict
        Dictionnary of key parameters/configs for our radar (w.r.t. config file) :
            - chirps per frame
            - number of Doppler bins
            - number of range bins
            - range resolution
            - maximum unambiguous range
            - Doppler resolution
            - maximum Doppler velocity

    Raises
    ------
    ValueError
        If required configuration sections are missing.

    """



    # ---- basic checks to prevent invalid config file parsing ----

    if cfg["frameCfg"] is None:
        raise ValueError("Missing frameCfg")

    if not cfg["profileCfg"]:
        raise ValueError("Missing profileCfg")

    if not cfg["chirpCfg"]:
        raise ValueError("Missing chirpCfg")
    


    # ---- Parse 'frameCfg', 'chirpCfg', 'profileId' (multiple) ----
     
    frame = cfg["frameCfg"]
    used_chirps = chirps_used_in_frame(cfg["chirpCfg"], frame)

    if not used_chirps:
        raise ValueError("No chirpCfg entries overlap with frameCfg chirp range")


    # ---- 'profileId' (multiple)

    profile_ids = sorted(set(chirp["profileId"] for chirp in used_chirps))

    if len(profile_ids) > 1:
        print("WARNING: Multiple profile IDs used in frame:", profile_ids)
        print("Using the first one for calculations.")

    profile = cfg["profileCfg"][profile_ids[0]]

    tx_mask = 0
    for chirp in used_chirps:
        tx_mask |= chirp["txEnable"]

    if tx_mask == 0 and cfg["channelCfg"] is not None:
        tx_mask = cfg["channelCfg"]["txChannelEn"]

    num_tx = bit_count(tx_mask)

    if num_tx == 0:
        raise ValueError("Could not determine NUM_TX")

    num_rx = None
    if cfg["channelCfg"] is not None:
        num_rx = bit_count(cfg["channelCfg"]["rxChannelEn"])


    # Extract / Assign Key Parameters to compute the specs at which our radar will operate

    start_chirp_tx  =   frame["chirpStartIdx"]
    end_chirp_tx    =   frame["chirpEndIdx"]
    chirp_loops     =   frame["numLoops"]

    start_freq      =   profile["startFreq"]          # GHz
    idle_time       =   profile["idleTime"]           # us
    ramp_end_time   =   profile["rampEndTime"]        # us
    freq_slope      =   profile["freqSlopeConst"]     # MHz/us
    adc_samples     =   profile["numAdcSamples"]
    sample_rate     =   profile["digOutSampleRate"]   # ksps


    chirps_per_frame  =   (end_chirp_tx - start_chirp_tx + 1) * chirp_loops
    num_doppler_bins  =   chirps_per_frame / num_tx
    num_range_bins    =   adc_samples


    range_resolution   =    (C * sample_rate * 1e3 / (2 * freq_slope * 1e12 * adc_samples))
    max_range          =    ( 300 * max_range_scale * sample_rate / (2 * freq_slope * 1e3) )
    doppler_resolution =    ( C / ( 2 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * num_doppler_bins * num_tx ) )
    max_doppler        =    ( C / ( 4 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * num_tx ) )


    return {
            "START_CHIRP_TX": start_chirp_tx,
            "END_CHIRP_TX": end_chirp_tx,
            "CHIRP_LOOPS": chirp_loops,
            "NUM_TX": num_tx,
            "NUM_RX": num_rx,
            "TX_MASK": tx_mask,
            "CHIRPS_PER_FRAME": chirps_per_frame,
            "NUM_DOPPLER_BINS": num_doppler_bins,
            "NUM_RANGE_BINS": num_range_bins,
            "ADC_SAMPLES": adc_samples,
            "SAMPLE_RATE": sample_rate,
            "FREQ_SLOPE": freq_slope,
            "START_FREQ": start_freq,
            "IDLE_TIME": idle_time,
            "RAMP_END_TIME": ramp_end_time,
            "RANGE_RESOLUTION": range_resolution,
            "MAX_RANGE": max_range,
            "DOPPLER_RESOLUTION": doppler_resolution,
            "MAX_DOPPLER": max_doppler,
            "FRAME_DURATION_MS": frame["framePeriodicity"],
    }




def print_params(params):
    """
    Prints extracted and computed radar setup parameters w.r.t. config file.

    Parameters
    ----------
    params : dict
        Dictionary returned by 'compute_params'.

    Returns
    -------
    None
        Function only prints in terminal.


    """


    print("-------- EXTRACTED PARAMETERS --------")
    print(f"START_CHIRP_TX:        {params['START_CHIRP_TX']}")
    print(f"END_CHIRP_TX:          {params['END_CHIRP_TX']}")
    print(f"CHIRP_LOOPS:           {params['CHIRP_LOOPS']}")
    print(f"NUM_TX:                {params['NUM_TX']}")
    print(f"NUM_RX:                {params['NUM_RX']}")
    print(f"TX_MASK:               {params['TX_MASK']}")
    print(f"ADC_SAMPLES:           {params['ADC_SAMPLES']}")
    print(f"SAMPLE_RATE:           {params['SAMPLE_RATE']} ksps")
    print(f"FREQ_SLOPE:            {params['FREQ_SLOPE']} MHz/us")
    print(f"START_FREQ:            {params['START_FREQ']} GHz")
    print(f"IDLE_TIME:             {params['IDLE_TIME']} us")
    print(f"RAMP_END_TIME:         {params['RAMP_END_TIME']} us")
    print(f"FRAME_DURATION:        {params['FRAME_DURATION_MS']} ms")


    print()
    print("-------- CALCULATED PARAMETERS --------")
    print(f"Chirps Per Frame:      {params['CHIRPS_PER_FRAME']}")
    print(f"Num Doppler Bins:      {params['NUM_DOPPLER_BINS']}")
    print(f"Num Range Bins:        {params['NUM_RANGE_BINS']}")
    print(f"Range Resolution:      {params['RANGE_RESOLUTION']:.6f} m")
    print(f"Max Unambiguous Range: {params['MAX_RANGE']:.6f} m")
    print(f"Doppler Resolution:    {params['DOPPLER_RESOLUTION']:.6f} m/s")
    print(f"Max Doppler:           {params['MAX_DOPPLER']:.6f} m/s")
    print("---------------------------------------")
    print()





def main():
    """
    Command-line entry point.

    This function:

        1. Reads the '.cfg' file path from the command line.
        2. Parses the configuration file.
        3. Computes radar parameters.
        4. Prints the results.

    Example
    -------
    Run from terminal:

        python compute_mmwave_cfg_params.py config.cfg

    Optional max range scale (by default, scale param is set to the value for the AWR1843 as specified in the TI document):

        python compute_mmwave_cfg_params.py config.cfg --max-range-scale 0.8

    Returns
    -------
    None
    """
    
    #   HOW TO USE :
    #
    #       python -m streaming.configure my_config.cfg
    #
    
    # ------- MAIN PARSER

    parser = argparse.ArgumentParser( description="Compute radar parameters from a TI mmWave .cfg file.")

    parser.add_argument("cfg_file", help="Path to the .cfg file")

    parser.add_argument("--max-range-scale", type=float, default=0.9, help="Scale factor used in max range formula. Default (AWR1843 limitation) : 0.9")

    args = parser.parse_args()


    # TODO : this is supposed to be the config file we use !
    cfg_file_path = "../configs/"+args.cfg_file  # NOTE : THIS IS THE PATH TO THE CONFIG FILE (equivalent to lua files)


    # ------ Operations on Config File

    # Parse config file
    cfg = parse_cfg(cfg_file_path)
    
    # Compute parameters w.r.t. config file
    params = compute_params(cfg, max_range_scale=args.max_range_scale)

    # Print computed parameters
    print_params(params)

    


if __name__ == "__main__":
    main()