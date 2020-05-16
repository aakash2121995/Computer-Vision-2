import numpy as np
import cv2


class PatchSampler():
    def __init__(self, train_images_list, gt_segmentation_maps_list, classes_colors, patchsize):

        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        self.class_colors = classes_colors
        self.patchsize = patchsize

    # Function for sampling patches for each cimg = cv2.imreadlass
    # provide your implementation
    # should return extracted patches with labels
    def extractpatches(self):
        patches = [[] for i in range(4)]
        for im, im_seg in zip(self.train_images_list,self.gt_segmentation_maps_list):
            img = cv2.imread("images/"+im)
            img_seg = cv2.imread("images/"+im_seg)

            for index in range(100000):
                i = np.random.randint(0,img_seg.shape[0]-self.patchsize)
                j = np.random.randint(0,img_seg.shape[1]-self.patchsize)
                patch = img_seg[ i:i+self.patchsize, j:j+self.patchsize]
                for num in self.class_colors:
                    if np.all(patch==num):
                        patches[num].append(img[ i:i+self.patchsize, j:j+self.patchsize])
        min_patch = min(len(patches[0]),len(patches[1]),len(patches[2]) ,len(patches[3]))
        patches = patches[0][:min_patch]+ patches[1][:min_patch]+ patches[2][:min_patch] + patches[3][:min_patch]
        labels = [0 for i in range(min_patch)]+ [1 for i in range(min_patch)] + [2 for i in range(min_patch)] + [3 for i in range(min_patch)]
        return np.stack(patches), np.array(labels)
    # feel free to add any helper functions


