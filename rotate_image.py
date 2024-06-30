import cv2
import numpy as np

def project_radar_coordinates_to_image(radar_x, radar_y, camera_matrix, distortion_coeffs, rotation_vec, translation_vec):
    # Create a 3D point from radar coordinates
    radar_point = np.array([[radar_x], [radar_y], [0]])

    # Project the 3D point onto the camera image
    image_points, _ = cv2.projectPoints(radar_point, rotation_vec, translation_vec, camera_matrix, distortion_coeffs)

    # Extract the pixel coordinates from the projected points
    pixel_x, pixel_y = image_points[0][0]

    return int(pixel_x), int(pixel_y)

# Specify the radar coordinates
radar_x = 10  # Adjust the radar X-coordinate as needed
radar_y = 5   # Adjust the radar Y-coordinate as needed

fx = 1546.7825 # focal length in x direction
fy = 1548.6482 # focal length in y direction
cx = 1043.1214 # x-coordinate of the principal point
cy = 592.6109 # y-coordinate of the principal point
k1 = -0.3747 # radial distortion coefficient
k2 = 0.1055 # radial distortion coefficient
p1 = 0.0 # tangential distortion coefficient
p2 = 0.0 # tangential distortion coefficient
k3 = 0.0 # radial distortion coefficient
# Specify the camera intrinsic parameters
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

distortion_coeffs = np.array([k1, k2, p1, p2, k3])

# Specify the camera extrinsic parameters (rotation and translation vectors)
rotation_vec = np.array([0.38302194, -0.34438426, -2.59991865])  # Adjust the rotation vector as needed
translation_vec = np.array([1.82262918, 1.77922866, 104.81224315])  # Adjust the translation vector as needed

# Call the function to project radar coordinates onto the camera image
pixel_x, pixel_y = project_radar_coordinates_to_image(radar_x, radar_y, camera_matrix, distortion_coeffs, rotation_vec, translation_vec)

# Display the result
print(f"Projected pixel coordinates: ({pixel_x}, {pixel_y})")
