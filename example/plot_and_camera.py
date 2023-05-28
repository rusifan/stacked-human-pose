import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# initialize the camera
cap = cv2.VideoCapture(0)

# create a figure and an Axes3D object for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# loop through the frames
while True:
    # read a frame from the camera
    ret, frame = cap.read()

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # do your pose estimation and obtain the 3D keypoints here
    keypoints = np.array(
                            [[ 0.00954536 ,-0.5248383  , 5.4809194 ],
                            [-0.11578162 ,-0.53698    , 5.4426193 ],
                            [-0.12744609 ,-0.1485115  , 5.3599124 ],
                            [-0.11126095 , 0.23789597 , 5.584008  ],
                            [ 0.13379143 ,-0.51052433 , 5.5140343 ],
                            [ 0.11097349 ,-0.1164324  , 5.548572  ],
                            [ 0.10632887 , 0.24657953 , 5.7272058 ],
                            [ 0.10203433 ,-0.9775337  , 5.285085  ],
                            [ 0.14218985 ,-1.0047085  , 5.2082024 ],
                            [ 0.13970663 ,-1.1293828  , 5.1460843 ],
                            [ 0.2378531  ,-0.91875756 , 5.356876  ],
                            [ 0.29758015 ,-0.6622696  , 5.47596   ],
                            [ 0.25525945 ,-0.46913373 , 5.297254  ],
                            [-0.08292694 ,-0.9436176  , 5.278473  ],
                            [-0.29142407 ,-0.7513262  , 5.3098354 ],
                            [-0.24949345 ,-0.48693958 , 5.3255987 ]]
                                                                    ) # for right hand
    skeleton = [
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9],
    [7, 10], [10, 11], [11, 12],
    [7, 13], [13, 14], [14, 15]
    ] 

    # clear the previous plot
    ax.cla()

    # plot the 3D keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2])
    for connection in skeleton:
        x = [keypoints[connection[0], 0], keypoints[connection[1], 0]]
        y = [keypoints[connection[0], 1], keypoints[connection[1], 1]]
        z = [keypoints[connection[0], 2], keypoints[connection[1], 2]]
        ax.plot(x, y, z, c='b')
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([4, 6])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=-79, azim=-90)
    # show the camera feed and the plot in one window
    cv2.imshow('Camera Feed', frame)
    plt.pause(0.001)
    plt.draw()

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
