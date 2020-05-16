from RandomForest import Forest
from Sampler import PatchSampler
import numpy as np
import cv2

train_images_list = []
gt_segmentation_maps_list = []
with open('/home/saba/Documents/Computer-Vision-2/Sheet02/images/train_images.txt', 'r') as f:
    no_img, no_seg_map = (f.readline().split())
    for i in range(int(no_img)):
        img, seg_img = f.readline().split()
        train_images_list.append(img)
        gt_segmentation_maps_list.append(seg_img)

# train_images_list =
# gt_segmentation_maps_list
classes_colors = [0, 1, 2, 3]
patchsize = 16

def main():
    sampler = PatchSampler(train_images_list, gt_segmentation_maps_list, classes_colors, patchsize)
    patches, labels = sampler.extractpatches()

if __name__ == "__main__":
    main()
# provide your implementation for the sheet 2 here


