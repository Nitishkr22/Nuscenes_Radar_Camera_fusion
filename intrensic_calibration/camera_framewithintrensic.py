import cv2
import numpy as np
import os

# Define the intrinsic parameters of the camera
# (these values depend on the camera and can be obtained from calibration)
fx = 1546.7825 # focal length in x direction
fy = 1548.6482 # focal length in y direction
cx = 1043.1214 # x-coordinate of the principal point
cy = 592.6109 # y-coordinate of the principal point
k1 = -0.3747 # radial distortion coefficient
k2 = 0.1055 # radial distortion coefficient
p1 = 0.0 # tangential distortion coefficient
p2 = 0.0 # tangential distortion coefficient
k3 = 0.0 # radial distortion coefficient

# Create the camera matrix and distortion coefficients from the intrinsic parameters
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# Create a VideoCapture object to capture frames from a camera
cap = cv2.VideoCapture(0) # 0 for the first/default camera, change if you have multiple cameras

# Define the directory to save the image files in
output_dir = 'frames3'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the frame rate (in frames per second) at which to save the frames to image files
frame_rate = 5 # 5 frames per second

# Initialize the frame counter
frame_count = 0

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # If the frame was successfully captured
    if ret:
        # Apply the distortion correction to the frame using the camera matrix and distortion coefficients
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Save the undistorted frame as an image file in the output directory
        file_name = os.path.join(output_dir, f"frame{frame_count:04d}.jpg")
        cv2.imwrite(file_name, undistorted_frame)

        # Increment the frame counter
        frame_count += 1

        # Display the frame in a window
        cv2.imshow('frame', frame)
        cv2.imshow('undistorted_frame', undistorted_frame)

    # Wait for a key press to exit the program
    if cv2.waitKey(1) == ord('q'):
        break

    # Wait for the specified amount of time before capturing the next frame
    delay = int(1000 / frame_rate)
    cv2.waitKey(delay)

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
