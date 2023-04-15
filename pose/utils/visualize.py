import numpy as np
from matplotlib import pyplot as plt


def show(image, heatmap, target):
    plt.figure(figsize=(12, 8))

    plt.subplot(131)
    plt.title('Heatmap')
    heatmap = heatmap.detach().cpu().numpy()
    plt.imshow(np.max(heatmap, axis=0))

    plt.subplot(132)
    plt.title('Target')
    target = target.detach().cpu().numpy()
    plt.imshow(np.max(target, axis=0))

    plt.subplot(133)
    plt.title('Predicted')
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.show()


def show_heatmap(heatmap, target):
    plt.figure(figsize=(20, 8))

    heatmap = heatmap.detach().cpu().numpy()
    for joint_i, joint_map in enumerate(heatmap):
        plt.subplot(2, 16, joint_i + 1)
        plt.imshow(joint_map)

    target = target.detach().cpu().numpy()
    for joint_i, joint_map in enumerate(target):
        plt.subplot(2, 16, 16 + joint_i + 1)
        plt.imshow(joint_map)
    plt.show()
