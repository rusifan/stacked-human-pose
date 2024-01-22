from __future__ import print_function, absolute_import

# from pose.datasets.human36_dataloader import hum36m_dataloader #16kps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
annotation_path_train = "annotation_body3d/fps5/h36m_train.npz"
annotation_path_test = "annotation_body3d/fps5/h36m_test.npz"
root_data_path = "/netscratch/nafis/human-pose/dataset_hum36m_10f1s/"

print("Loading the numpy data...")
train_data = np.load(root_data_path+annotation_path_train, allow_pickle=True)
test_data = np.load(root_data_path+annotation_path_test, allow_pickle=True)

train_kp3d = np.array(train_data['S'])
test_kp3d = np.array(test_data['S'])

train_kp3d = train_kp3d.reshape(-1, 3)
test_kp3d = test_kp3d.reshape(-1, 3)

#plot the z axis distribution
plt.hist(train_kp3d[:,2], bins=100)
plt.title("distribution of depth of 3d keypoints in train set")
plt.xlabel("distance from camera")
plt.ylabel("frequency")
plt.savefig("train_z.png")


#clear the plot
plt.clf()
plt.hist(test_kp3d[:,2], bins=100)
plt.title("distribution of depth of 3d keypoints in test set")
plt.xlabel("distance from camera")
plt.ylabel("frequency")
plt.savefig("test_z.png")

