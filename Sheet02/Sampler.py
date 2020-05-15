import numpy as np
import cv2


class PatchSampler():
    def __init__(self, train_images_list, gt_segmentation_maps_list, classes_colors, patchsize):

        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        self.class_colors = classes_colors
        self.patchsize = patchsize

    # Function for sampling patches for each class
    # provide your implementation
    # should return extracted patches with labels
    def extractpatches(self):
        pass

    # feel free to add any helper functions


