import cv2
import os

# Create a VideoCapture object to capture frames from a camera
cap = cv2.VideoCapture(0) # 0 for the first/default camera, change if you have multiple cameras

# Define the directory to save the image files in
output_dir = 'frames'
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
        # Save the frame as an image file in the output directory
        file_name = os.path.join(output_dir, f"frame{frame_count:04d}.jpg")
        cv2.imwrite(file_name, frame)
        # Increment the frame counter
        frame_count += 1

        # Display the frame in a window
        cv2.imshow('frame', frame)

    # Wait for a key press to exit the program
    if cv2.waitKey(1) == ord('q'):
        break

    # Wait for the specified amount of time before capturing the next frame
    delay = int(1000 / frame_rate)
    cv2.waitKey(delay)

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
import cv2
import os

# Create a VideoCapture object to capture frames from a camera
cap = cv2.VideoCapture(0) # 0 for the first/default camera, change if you have multiple cameras

# Define the directory to save the image files in
output_dir = 'frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the frame rate (in frames per second) at which to save the frames to image files
frame_rate = 30 # 5 frames per second

# Initialize the frame counter
frame_count = 0

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # If the frame was successfully captured
    if ret:
        # Save the frame as an image file in the output directory
        file_name = os.path.join(output_dir, f"frame{frame_count:04d}.jpg")
        cv2.imwrite(file_name, frame)
        # Increment the frame counter
        frame_count += 1

        # Display the frame in a window
        cv2.imshow('frame', frame)

    # Wait for a key press to exit the program
    if cv2.waitKey(1) == ord('q'):
        break

    # Wait for the specified amount of time before capturing the next frame
    delay = int(1000 / frame_rate)
    cv2.waitKey(delay)

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
