import cv2

# Read the image
img = cv2.imread('/home/radar/Documents/camera/frames2/frame0122.jpg')

# Find the non-zero pixels
non_zero_pixels = cv2.findNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
imga = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Extract the u and v values from the non-zero pixels and store them in a tuple
u_values = tuple(non_zero_pixels[:, 0, 0])
v_values = tuple(non_zero_pixels[:, 0, 1])

# Print the u and v values
# print("u values:",u_values)
# print("v values:",v_values)
print(imga[:,1])