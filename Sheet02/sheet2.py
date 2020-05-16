from RandomForest import Forest
from Sampler import PatchSampler
import numpy as np
import cv2
from Tree import DecisionTree

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

train_images_list = []
gt_segmentation_maps_list = []
with open('images/train_images.txt', 'r') as f:
    no_img, no_seg_map = (f.readline().split())
    for i in range(int(no_img)):
        img, seg_img = f.readline().split()
        train_images_list.append(img)
        gt_segmentation_maps_list.append(seg_img)

test_images_list = []
test_segmentation_maps_list = []
with open('images/test_images.txt', 'r') as f:
    no_img, no_seg_map = (f.readline().split())
    for i in range(int(no_img)):
        img, seg_img = f.readline().split()
        test_images_list.append(img)
        test_segmentation_maps_list.append(seg_img)

# train_images_list =
# gt_segmentation_maps_list
classes_colors = [0, 1, 2, 3]
patchsize = 16

def main():
    sampler = PatchSampler(train_images_list, gt_segmentation_maps_list, classes_colors, patchsize)
    patches, labels = sampler.extractpatches()

    tree_params = {'depth':15,'pixel_locations':100, 'random_color_values':10,'no_of_thresholds':50,'minimum_patches_at_leaf':20,'classes':[0,1,2,3]}
    tree = DecisionTree(patches,labels,tree_params)
    tree.train()

    for im, im_seg in zip(test_images_list, test_segmentation_maps_list):
        img = cv2.imread("images/" + im)
        img_seg = cv2.imread("images/" + im_seg)
        op = tree.predict(img)
        display_image(f'Predicted Labels for{im}',op)
        display_image(f'Actual Labels for{im}',img_seg)
        # print((op==img_seg).sum()/(op.shape[0]*op.shape[1]))

    print('tree')

if __name__ == "__main__":
    main()
# provide your implementation for the sheet 2 here


