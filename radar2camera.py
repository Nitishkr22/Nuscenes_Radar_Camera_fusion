import cv2
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def project_radar_on_camera(camera_image_path, radar_distance, azimuth_angle):
#     # Load camera image
#     camera_image = cv2.imread(camera_image_path)

#     # Calculate radar coordinates
#     radar_x = radar_distance * math.cos(math.radians(azimuth_angle))
#     radar_y = radar_distance * math.sin(math.radians(azimuth_angle))

#     # Calculate the corresponding point on the camera image
#     camera_x = int(camera_image.shape[1] / 2 + radar_x)
#     camera_y = int(camera_image.shape[0] / 2 + radar_y)

#     # Draw a circle on the camera image at the projected point
#     cv2.circle(camera_image, (camera_x, camera_y), 5, (0, 0, 255), -1)

#     # Display the camera image with the projected point
#     cv2.imshow("Radar Projection", camera_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Example usage
# camera_image_path = "/home/radar/Downloads/frame0157.jpg"
# radar_distance = 2.81075  # Example radar distance in meters
# azimuth_angle = -6.0504 # Example azimuth angle in degrees

# project_radar_on_camera(camera_image_path, radar_distance, azimuth_angle)

############
rdf = pd.read_csv("/home/radar/Downloads/radar_data.csv")

gdf = rdf.groupby('timestamp')
# print(gdf)
for timestamp, group in gdf:
    # print(len(group))
    # print(timestamp)
    fx = np.array([])
    fy = np.array([])
    # # Do something with the group of rows with the same timestamp
    radial_distance = group[' Radial Distance'].tolist()
    # print(radial_distance)
    azi = group[' azimuth hypothesis'].tolist()
    for i in range(0, len(radial_distance)):
    #   if(i >= 0 and i < len(radial_distance)and len(radial_distance) == len(azi) and radial_distance[i] >= 0 and radial_distance[i] < 5):
        fx = np.append(fx, radial_distance[i]*np.cos(azi[i]))
        fy = np.append(fy, radial_distance[i]*np.sin(azi[i]))
    # print(type(fx.tolist()))
    # print(type(fy))


    # plt.scatter(fx.tolist(), fy.tolist())
    # plt.show()
    #   else:
    #     continue
    # if(len(fx) == len(fy) and len(fx) > 0):
    # plt.figure()
    # plt.scatter(fx,fy)
    #   print(len(fx))
    # if(radial_distance)
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax.scatter(azi, radial_distance)

    # # Set the origin to the north
    # ax.set_theta_zero_location("N")

    # # Set the theta direction clockwise
    # ax.set_theta_direction(-1)

    # # Set the radial limits
    # ax.set_rlim(0, max(radial_distance))

    # # Set grid lines
    # ax.grid(True)

    # # Show the plot
    # plt.show()

radar_data_list = [
    {'distance': 10.0, 'azimuth': 45.0},
    {'distance': 15.0, 'azimuth': -30.0},
    {'distance': 8.0, 'azimuth': 90.0}
]
print(radar_data_list)