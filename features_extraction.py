import cv2
import numpy as np

# Load the image
img = cv2.imread("/home/radar/Documents/camera/frames2/frame0002.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Harris corner detection to find keypoints in the image
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

# Threshold the corner response to extract the keypoints
thresh = 0.01 * dst.max()
keypoints = []
for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
        if dst[i,j] > thresh:
            keypoints.append(cv2.KeyPoint(j,i,3))

# Extract SIFT descriptors for the keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.compute(gray, keypoints)

# Simulate radar data for the keypoints
radar_data = np.zeros((len(keypoints), 2))
for i in range(len(keypoints)):
    x, y = keypoints[i].pt
    radar_data[i,0] = x + np.random.normal(scale=5)
    radar_data[i,1] = y + np.random.normal(scale=5)

# Match the keypoints based on their Euclidean distance in the radar data
matches = []
for i in range(len(keypoints)):
    best_match = None
    best_distance = float('inf')
    for j in range(len(keypoints)):
        distance = np.sqrt((radar_data[i,0]-radar_data[j,0])**2 + (radar_data[i,1]-radar_data[j,1])**2)
        if distance < best_distance:
            best_match = j
            best_distance = distance
    if best_match is not None:
        matches.append(cv2.DMatch(i,best_match,0))

# Draw the matched keypoints on the image
img_matches = cv2.drawMatches(img, keypoints, img, keypoints, matches, None)
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
