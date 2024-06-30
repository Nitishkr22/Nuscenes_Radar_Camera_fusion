import cv2
import numpy as np
import glob
import json

# Define the calibration pattern size
pattern_size = (9, 6)

# Define the calibration pattern object points
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Create arrays to store object points and image points from all images
obj_points = []  # 3D points in the real world
img_points = []  # 2D points in image plane

# Load calibration images
images = glob.glob('/home/radar/Documents/camera/image_checker_webcam/*.jpg')

# Iterate through all images
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If corners are found, add object points and image points
    if ret == True:
        obj_points.append(objp)
        img_points.append(corners)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Compute homography matrix
homography, mask = cv2.findHomography(objp[:, :2], img_points[0])

# Save homography matrix to JSON file
homography_data = {'homography_matrix': homography.tolist()}
with open('homography_matrix.json', 'w') as f:
    json.dump(homography_data, f)
