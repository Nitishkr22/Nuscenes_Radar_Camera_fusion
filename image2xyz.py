import cv2

def get_coordinates(image_path, click_event):
    # Read the image
    image = cv2.imread(image_path)

    # Create a window to display the image
    cv2.namedWindow('Image')
    
    # Set the callback function for mouse events
    cv2.setMouseCallback('Image', click_event)

    # Display the image
    cv2.imshow('Image', image)
    
    # Wait for a key press
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates: ({x}, {y})")

# Specify the path to your image
image_path = '/home/radar/Documents/camera/frames3/frame0013.jpg'

# Call the function to get the coordinates
get_coordinates(image_path, click_event)
