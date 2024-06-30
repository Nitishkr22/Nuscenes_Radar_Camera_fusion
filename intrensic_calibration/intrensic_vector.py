import cv2
import numpy as np

def calibrate_camera(image_paths, pattern_size, square_size):
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (7,5,0)
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images
    object_points_list = []  # 3D points in real world space
    image_points_list = []  # 2D points in image plane

    # Iterate through all calibration images
    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If corners are found, add object points and image points
        if ret:
            object_points_list.append(object_points)
            image_points_list.append(corners)

    # Perform camera calibration
    ret, camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(
        object_points_list, image_points_list, gray.shape[::-1], None, None
    )

    return camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors

# Specify the paths to calibration images
image_paths = [
    '/home/radar/Documents/camera/image_checker_webcam/WIN_20230428_08_34_01_Pro.jpg',
    '/home/radar/Documents/camera/image_checker_webcam/WIN_20230428_08_34_48_Pro.jpg',
    '/home/radar/Documents/camera/image_checker_webcam/WIN_20230428_08_34_19_Pro.jpg',
    # Add more calibration images as needed
]

# Specify the size of the calibration pattern (e.g., number of inner corners)
pattern_size = (9, 6)  # Adjust according to your calibration pattern

# Specify the size of the squares in the calibration pattern (in the same units as the coordinates)
square_size = 25  # Adjust according to your calibration pattern

# Call the function to calibrate the camera
camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors = calibrate_camera(
    image_paths, pattern_size, square_size
)

# Print the rotation and translation vectors
for i in range(len(rotation_vectors)):
    print(f"Rotation Vector {i+1}: {rotation_vectors[i]}")
    print(f"Translation Vector {i+1}: {translation_vectors[i]}")
