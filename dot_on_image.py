import numpy as np
import cv2

# Create a black image
# img = np.zeros((512, 512, 3), np.uint8)
img = cv2.imread("/home/radar/Documents/camera/frames2/frame0111.jpg")

# Draw a red dot at position (100, 100)
cv2.circle(img, (100, 100), 4, (0, 255, 255), -1)

# Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
