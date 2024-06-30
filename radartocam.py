import numpy as np

# Mounting coordinates of radar and camera with respect to vehicle
radar_mount = np.array([x_r, y_r, z_r]) # in meters
camera_mount = np.array([x_c, y_c, z_c]) # in meters

# Intrinsic parameters of camera
fx, fy = focal_length_x, focal_length_y
cx, cy = principal_point_x, principal_point_y

# Frame-wise radar and camera detections
radar_detections = [...] # list of (x_r, y_r, z_r) tuples
camera_detections = [...] # list of (u, v) tuples

# Function to convert pixel coordinates to camera coordinates
def pixel2cam(u, v, fx, fy, cx, cy, depth):
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return x, y, z

# Function to find transformation between radar and camera
def calibrate(radar_mount, camera_mount, fx, fy, cx, cy, radar_detections, camera_detections):
    # Convert camera detections to camera coordinates
    camera_coords = []
    for u, v in camera_detections:
        x, y, z = pixel2cam(u, v, fx, fy, cx, cy, depth)
        camera_coords.append([x, y, z])
    camera_coords = np.array(camera_coords)

    # Convert radar detections to radar coordinates
    radar_coords = np.array(radar_detections)

    # Calculate transformation between radar and camera
    M_r2v = np.eye(4)
    M_r2v[:3, 3] = radar_mount
    M_c2v = np.eye(4)
    M_c2v[:3, 3] = camera_mount
    M_r2c = np.linalg.inv(M_c2v) @ M_r2v
    H, _ = cv2.findHomography(camera_coords, radar_coords)

    # Combine transformation with homography to get final transformation
    M_c2r = np.linalg.inv(H) @ M_r2c[:3, :4]
    
    return M_c2r

# Call calibration function with your specific values
M_c2r = calibrate(radar_mount, camera_mount, fx, fy, cx, cy, radar_detections, camera_detections)
