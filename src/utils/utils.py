import numpy as np
from scipy.interpolate import RegularGridInterpolator



def normalize_for_display(power, exponent=8):
    """
    Normalize a radar power map for plotting.
    """

    power = np.abs(power).astype(np.float32)

    max_val = np.max(power)
    if max_val > 0:
        power = power / max_val

    return power ** exponent



def radar_power_on_cartesian_grid(power, azimuth_bins_deg, range_bins_m, radar_pose, x_grid, y_grid):
    """
    Convert one radar's polar (azimuth-range) power map onto a global Cartesian grid.
    
    Parameters
    ----------
    power : array_like, shape (num_azimuth, num_range)
        Beamformed power map.
    azimuth_bins_deg : array_like, shape (num_azimuth,)
        Azimuth bins in degrees. 0 deg points perpendicular to the radar board .
    range_bins_m : array_like, shape (num_range,)
        Range-bin centers in meters.
    radar_pose : tuple
        ``(tx, ty, yaw_deg)`` pose of the radar in the global frame, where
        global x is forward and global y is left/right.
    x_grid, y_grid : array_like
        Global Cartesian meshgrid arrays.


    Returns
    -------
    cart_power : ndarray
        Power map on the supplied global Cartesian grid.

    """

    # Extract radar pose parameters (x-y offsets and direction phase offset)
    tx, ty, yaw_deg = radar_pose
    yaw = np.deg2rad(yaw_deg)


    # Global point relative to radar position
    dx = x_grid - tx
    dy = y_grid - ty


    # Rotate global cartesian coordinates into radar-local cartesian coordinates (IDEA : we put ourselves in the radar point of view)
    x_local = np.cos(yaw) * dx + np.sin(yaw) * dy
    y_local = -np.sin(yaw) * dx + np.cos(yaw) * dy


    # Convert local Cartesian coordinates to Polar coordinates (IDEA : we change our coordinate system to that of the measurements made by the radar)
    r_radar = np.sqrt(x_local**2 + y_local**2)
    phi_radar = np.rad2deg(np.arctan2(y_local, x_local))


    #   INTERPOLATE POWER VALUES 
    #   ------------------------
    #     for our CARTESIAN GRID (using corresponding mapped polar coordinates (computed above)) 
    #     FROM THE "ACTUAL" RADAR POWER MEASUREMENTS (in its own "local" view (irrespective of our global view))
    

    # Interpolator expects points as (phi, range)       --   NOTE TO SELF : we find the closest angle bin and closest range bin
    interpolator = RegularGridInterpolator((azimuth_bins_deg, range_bins_m), power, bounds_error=False, fill_value=0.0)


    # Create pairs of phi-range coordinate (from the radar polar phi/range values we found by mapping global cartesian coordinates to polar ones)
    #   we use these "radar localized" polar coordinates to interpolate power values on the final cartesian grid from the actual power measurements
    #   of the radar at "actual polar axis locations".
    
    query_points = np.stack([phi_radar.ravel(), r_radar.ravel()], axis=-1)


    #  to illustrate the proceedure below, what we do is :
    #   - Look inside power1 around phi (i.e. 12.4°) and range (i.e. 8.7 m)
    #   - Interpolate between nearby cells
    #   - Return the estimated power value

    cart_power = interpolator(query_points).reshape(x_grid.shape)


    return cart_power






#def radar_power_on_cartesian_grid(
#    power,
#    azimuth_bins_deg,
#    range_bins_m,
#    radar_pose,
#    x_grid,
#    y_grid,
#):
#    """
#    Project one radar's azimuth-range power map onto the shared Cartesian grid.
#
#    The beamformer produces a polar image indexed by:
#        - azimuth angle, where 0 deg means straight ahead
#        - radar range bin, in meters
#
#    This function always uses a range-axis-rectified projection. The plotted
#    forward coordinate is the radar range bin itself, not the true down-range
#    distance ``range * cos(azimuth)``. Therefore a horizontal / constant-range
#    row in the polar map becomes a straight constant-x line in the Cartesian
#    display. This keeps the Cartesian view directly aligned with the polar
#    range axis.
#
#    Parameters
#    ----------
#    power : array_like, shape (num_azimuth, num_range)
#        Beamformed power map.
#    azimuth_bins_deg : array_like, shape (num_azimuth,)
#        Azimuth bins in degrees. 0 deg points along the radar boresight.
#    range_bins_m : array_like, shape (num_range,)
#        Range-bin centers in meters.
#    radar_pose : tuple
#        ``(tx, ty, yaw_deg)`` pose of the radar in the global frame, where
#        global x is forward and global y is left/right.
#    x_grid, y_grid : array_like
#        Global Cartesian meshgrid arrays.
#    Returns
#    -------
#    cart_power : ndarray
#        Power map on the supplied global Cartesian grid.
#    """
#
#    power = np.asarray(power, dtype=np.float32)
#    azimuth_bins_deg = np.asarray(azimuth_bins_deg, dtype=np.float32)
#    range_bins_m = np.asarray(range_bins_m, dtype=np.float32)
#
#    if power.shape != (azimuth_bins_deg.size, range_bins_m.size):
#        raise ValueError(
#            "power must have shape "
#            f"(len(azimuth_bins_deg), len(range_bins_m)); got {power.shape}, "
#            f"expected {(azimuth_bins_deg.size, range_bins_m.size)}"
#        )
#
#    # RegularGridInterpolator requires strictly ascending axes. Keep the power
#    # array aligned if a future caller passes descending bins.
#    if np.any(np.diff(azimuth_bins_deg) <= 0):
#        order = np.argsort(azimuth_bins_deg)
#        azimuth_bins_deg = azimuth_bins_deg[order]
#        power = power[order, :]
#
#    if np.any(np.diff(range_bins_m) <= 0):
#        order = np.argsort(range_bins_m)
#        range_bins_m = range_bins_m[order]
#        power = power[:, order]
#
#    # Extract radar pose parameters. yaw rotates the radar-local forward axis
#    # into the global x/y frame.
#    tx, ty, yaw_deg = radar_pose
#    yaw = np.deg2rad(yaw_deg)
#
#    # Global point relative to radar position.
#    dx = x_grid - tx
#    dy = y_grid - ty
#
#    # Rotate global Cartesian coordinates into radar-local Cartesian
#    # coordinates: x_local is boresight/forward, y_local is lateral.
#    x_local = np.cos(yaw) * dx + np.sin(yaw) * dy
#    y_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
#
#    # Convert local Cartesian coordinates into the polar coordinates used to
#    # sample the measured azimuth-range map.
#    phi_radar = np.rad2deg(np.arctan2(y_local, x_local))
#
#    # Range-axis-rectified view: constant polar range stays at constant local
#    # forward coordinate, so the Cartesian display straightens the polar map
#    # with respect to the range axis.
#    r_radar = x_local
#
#    interpolator = RegularGridInterpolator(
#        (azimuth_bins_deg, range_bins_m),
#        power,
#        bounds_error=False,
#        fill_value=0.0,
#    )
#
#    query_points = np.stack([phi_radar.ravel(), r_radar.ravel()], axis=-1)
#    cart_power = interpolator(query_points).reshape(x_grid.shape)
#
#    # Remove anything behind the radar. Interpolation will usually zero these
#    # because the range becomes negative in rectified mode or the azimuth is
#    # out-of-bounds in physical mode, but the mask makes the convention explicit.
#    cart_power = np.where(x_local >= 0.0, cart_power, 0.0)
#
#    return cart_power.astype(np.float32, copy=False)





import numpy as np


def get_ant_pos_1d(num_x_stp, num_rx):
    """
    Computes the antenna positions for a 1D radar setup.

    Parameters
    ----------
    num_x_stp : int
        Number of steps in the x-direction.
    num_rx : int
        Number of receive antennas.

    Returns
    -------
    ant_pos : np.ndarray
        Array of antenna positions in the x-direction.
    """

    # Calculate the number of steps for each receiver
    num_x_stp_ = num_x_stp // num_rx

    # define the antenna spacing
    lm = 3e8/77e9 # define lambda for the antenna spacing

    # this is the receiver positions
    rx_pos = np.reshape(np.arange(1,num_rx+1,dtype=float),(-1,1)) * -lm / 2

    # this is the locations of the locations of the radar (we are moving it by lambda) 
    x_pos = (np.reshape(np.arange(1,num_x_stp_+1,dtype=float),(-1,1))) * lm

    # antenna positions for all receivers in the entire scan. /lm so that we don't have two factors of lm when we multiply them
    ant_pos = np.reshape(np.array([rx_pos + x_pos[i] for i in range(len(x_pos))]),(-1,1))
    ant_pos = ant_pos - ant_pos[0] # make sure first location is 0

    return ant_pos

def get_ant_pos_2d(num_x_stp, num_z_stp, num_rx):
    """
    Computes the antenna positions for a 2D radar setup.

    Parameters
    ----------
    num_x_stp : int
        Number of steps in the x-direction.
    num_z_stp : int
        Number of steps in the z-direction.
    num_rx : int
        Number of receive antennas.

    Returns
    -------
    x_ant_pos : np.ndarray
        Array of x-positions of the antennas.
    """

    num_x_stp_ = num_x_stp // num_rx

    lm = 3e8/77e9 # define lambda for the antenna spacing
    stp_size = 300*lm/4/369 # step size in the z (vertical) direction
    rx_pos = np.reshape(np.arange(1,num_rx+1,dtype=float),(-1,1)) * -lm / 2 # receiver positions 
    x_pos = (np.reshape(np.arange(0,num_x_stp_,dtype=float),(-1,1)) * lm).T # x (horizontal) positions of the radar
    x_ant_pos = np.reshape(np.squeeze(np.array([rx_pos + x_pos[0,i] for i in range(x_pos.shape[1])])),(-1,1)) # complete position of every receiver antenna

    # make it 0 indexed
    rx_pos = rx_pos - rx_pos[0] 
    x_pos = x_pos - x_pos[0,0]
    x_ant_pos = x_ant_pos - x_ant_pos[0]

    # z (vertical) positions defined
    z_pos = (np.reshape(np.arange(1,num_z_stp+1,dtype=float),(-1,1)) * stp_size).T
    z_pos = z_pos - z_pos[0,0]
    return x_ant_pos, z_pos, x_pos

def get_ant_static_2d(num_frames, num_tx, num_rx, adc_samples):
    """
    Computes virtual antenna positions for static radar setup.

    Parameters:
    ----------
        num_frames: number of frames
        num_tx: number of transmit antennas
        num_rx: number of receive antennas
        adc_samples: number of ADC samples per chirp

    Returns:
    -------
        x_ant_pos: np.ndarray of virtual antenna x-positions
        z_ant_pos: np.ndarray of virtual antenna z-positions
    """

    lm = 3e8 / 77e9  # lambda for 77 GHz

    RX_X = np.array([-3*lm/2, -lm, -lm/2, 0])      # 4 Rx
    RX_Z = np.array([0, 0, 0, 0])
    TX_X = np.array([0, lm, 2*lm])                 # 3 Tx
    TX_Z = np.array([0, lm/2, 0])

    x_ant_pos = []
    z_ant_pos = []

    for tx_i in range(num_tx):
        for rx_i in range(num_rx):
            x = TX_X[tx_i] + RX_X[rx_i]
            z = TX_Z[tx_i] + RX_Z[rx_i]
            x_ant_pos.append(x)
            z_ant_pos.append(z)

    x_ant_pos = np.array(x_ant_pos)
    z_ant_pos = np.array(z_ant_pos)

    # Optional: make origin zero-centered
    x_ant_pos -= np.min(x_ant_pos)
    z_ant_pos -= np.min(z_ant_pos)

    return x_ant_pos, z_ant_pos

# Helper function to get point cloud values
def plot_3d_cart_heatmap(ax,voxel,xaxis,yaxis,zaxis,threshold):
    ''''
    Returns X,Y,Z positions of voxels with power above a threshold.

    Parameters:
    - ax: matplotlib 3D axis to plot on
    - xaxis: x-values (for BF azimuth angles, for MF x distances)
    - yaxis: y-values (for BF elevation angles, for MF y distances)
    - zaxis: z-values (for range bins, for MF z distances)
    
    Returns:
    - X_: x points
    - Y_: y points
    - Z_: z points
    - intesn: intensity of the points (used for coloring)
    '''

    thresh = np.max(np.abs(voxel)) * threshold
    
    # Find indices where voxel values exceed the threshold
    ptcloud_lim = thresh
    pc_idx = np.where(voxel > ptcloud_lim)
    print(len(pc_idx))

    # Convert indices to subscripts
    x_idx, y_idx, z_idx = pc_idx[0],pc_idx[1],pc_idx[2]
    # Extract corresponding coordinates
    X_ = xaxis[x_idx]
    Y_ = yaxis[y_idx]
    Z_ = zaxis[z_idx]
    intesn = voxel[x_idx, y_idx, z_idx] 

    ax.scatter(X_,Y_, Z_, c=intesn, cmap='jet', marker='o')
    ax.view_init(elev=45, azim=45)  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Add a grid and make it interactive (movable)
    ax.grid(True)

    # return X_, Y_, Z_, intesn 


def load_raw_data(data_path):
    """
    Load raw radar data from a .mat file.

    Parameters:
    ----------
    data_path : str
        Path to the .mat file containing the raw radar data.

    Returns:
    -------
    radar_params : dict
        Dictionary containing radar parameters such as sample rate, number of samples, etc.
    raw_data : np.ndarray
        The raw radar data reshaped to (num_x_stp, num_z_stp, adc_samples).
    """

    import scipy.io as sio
    mat_data = sio.loadmat(data_path)

    raw_data = mat_data['data_raw']  # shape: (frames, tx, rx, samples)

    num_frames, num_tx, num_rx, adc_samples = raw_data.shape
    num_x_stp = num_tx * num_rx
    num_z_stp = num_frames

    # Reshape to (num_x_stp, num_z_stp, adc_samples)
    raw_data = raw_data.transpose(1, 2, 0, 3)  # (tx, rx, frames, samples)
    raw_data = raw_data.reshape(num_tx * num_rx, num_frames, adc_samples)

    radar_params = {
        'sample_rate': 10e6,
        'num_samples': adc_samples,
        'slope': 70.150e12,
        'lm': 3e8 / 77e9,
        'num_x_stp': num_x_stp,
        'num_z_stp': num_z_stp,
        'num_tx': num_tx,
        'num_rx': num_rx,
        'adc_samples': adc_samples,
        'num_frames': num_frames
    }

    return radar_params, raw_data


def sph2cart(az, el, r):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters:
    ----------
    az : array_like
        Azimuthal angle in radians.
    el : array_like
        Polar angle in radians.
    r : array_like
        Radius (distance from the origin).

    Returns:
    ----------
    x : array_like
        x-coordinate in Cartesian coordinates.
    y : array_like
        y-coordinate in Cartesian coordinates.
    z : array_like
        z-coordinate in Cartesian coordinates.
    """

    y = r * np.sin(el)
    rcosel = r * np.cos(el)
    x = rcosel * np.cos(az)
    z = rcosel * np.sin(az)
    return x, y, z


# Function to plot a 2D heatmap in polar coordinates
def plot_2d_heatmap(ax, data, theta, r, vmin=0, vmax=0.1):
    """
    Plot a 2D heatmap in polar coordinates.

    Parameters:
    ----------
        data: 2D numpy array
            The heatmap data to be plotted. Of size (theta x r)
        r_max: float
            Maximum radius of the polar plot.
    """

    R, Theta  = np.meshgrid(r,theta)

    ax.pcolormesh(Theta, R, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlim(theta[0],theta[-1])
    ax.set_ylim(r[0],r[-1])
    ax.grid(False)

# Function to plot a 2D heatmap in polar coordinates
def plot_2d_polar_heatmap(ax, data, az, el, vmin=0, vmax=0.1):
    """
    Plot a 2D heatmap in polar coordinates.

    Parameters:
    ----------
        data: 2D numpy array
            The heatmap data to be plotted.
        r_max: float
            Maximum radius of the polar plot.
    """

    # Create the heatmap
    ax.pcolormesh(az, el, data.T, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)

    # Label axes
    ax.set_xlabel(r"$\theta$ (Azimuthal Angle, radians)")
    ax.set_ylabel(r"$\phi$ (Polar Angle, radians)")
    ax.grid(False)
    ax.title.set_text("2D Polar Heatmap (φ-θ)")

# Function to plot a 3D polar heatmap as a point cloud
def plot_3d_polar_heatmap(ax, data, az, el,r,threshold):
    """
    Plot a 3D heatmap in spherical coordinates as a point cloud.

    Parameters:
    ----------
        data: 3D numpy array
            The heatmap data to be plotted. Should have shape (n_r, n_phi, n_theta).
        r_max: float
            Maximum radius of the spherical coordinates.
    """

    # Create a meshgrid of spherical coordinates
    R, Phi, Theta = np.meshgrid(r, az, el, indexing='ij')

    # Convert spherical coordinates to Cartesian for plotting
    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)

    # Flatten arrays for point cloud
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    values = data.flatten()
    thresh = np.max(np.abs(values)) * threshold
    
    # Find indices where voxel values exceed the threshold
    ptcloud_lim = thresh
    pc_idx = np.where(values > ptcloud_lim)

    # Convert indices to subscripts
    idx = pc_idx[0]
    # Extract corresponding coordinates
    X_ = x[idx]
    Y_ = y[idx]
    Z_ = z[idx]
    intesn = idx 

    # Plot the point cloud
    ax.scatter(X_, Y_, Z_, c=intesn, cmap='jet', s=10)
    ax.title.set_text("3D Polar Heatmap Point Cloud")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def cart2pol(x_flat, y_flat):
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x_flat : np.ndarray
        Array of x-coordinates.
    y_flat : np.ndarray
        Array of y-coordinates.

    Returns
    -------
    np.ndarray
        Array of polar coordinates in the form of (phi, r), where:
        - phi is the azimuthal angle in radians.
        - r is the radial distance from the origin.
    """

    phi_flat = np.arctan2(y_flat, x_flat)
    r_flat = np.hypot(x_flat, y_flat)

    return np.column_stack((phi_flat, r_flat))



