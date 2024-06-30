import cv2
import numpy as np
import glob

# Define the dimensions of the chessboard used in calibration
board_size = (10, 7) # number of corners in width and height

# Create arrays to store the object points (3D points in real world space) and image points (2D points in image space) for all calibration images
object_points = []
image_points = []

# Define the real world coordinates of the corners of the chessboard (assuming the chessboard lies in the xy-plane with z=0)
object_points_0 = np.zeros((board_size[0]*board_size[1], 3), np.float32)
object_points_0[:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2)

# Load the chessboard images from a folder
image_paths = glob.glob('/home/radar/Desktop/CAMERA_CALIBRATION_IMAGES/CENTER_CAM/*.png')

for image_path in image_paths:
    # Load the image from file
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in the grayscale image
    ret, corners = cv2.findChessboardCorners(gray, board_size)

    # If the corners were found
    if ret:
        # Draw the corners on the image and display it
        cv2.drawChessboardCorners(image, board_size, corners, ret)
        cv2.imshow('image', image)

        # Wait for a key press to capture the image or exit the program
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        if key == ord('c'):
            # Refine the corner positions to subpixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Add the object points and image points to the lists
            object_points.append(object_points_0)
            image_points.append(corners_refined)

# Calibrate the camera using the object points and image points
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# Print the intrinsic parameters of the camera
print('Camera matrix:\n', camera_matrix)
print('Distortion coefficients:\n', dist_coeffs)
