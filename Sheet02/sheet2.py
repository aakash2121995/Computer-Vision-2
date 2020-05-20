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
classes_colors = [[0,0,0], [0,0,255],[255,0,0],[0,255,0]]
patchsize = 16

def get_coloured_imgs(img):
    colored_op = np.zeros((img.shape[0], img.shape[1], 3))
    for cls, color in enumerate(classes_colors):
        colored_op[np.where(img == cls)] = color

    return colored_op

def main():
    sampler = PatchSampler(train_images_list, gt_segmentation_maps_list, [0,1,2,3], patchsize)
    patches, labels = sampler.extractpatches()

    tree_params = {'depth':15,'pixel_locations':100,
                   'random_color_values':10,'no_of_thresholds':50,
                   'minimum_patches_at_leaf':20,'classes':[0,1,2,3]}

    tree = DecisionTree(patches,labels,tree_params)
    tree.train()

    for im, im_seg in zip(test_images_list, test_segmentation_maps_list):
        img = cv2.imread("images/" + im)
        img_seg = cv2.imread("images/" + im_seg)
        op = tree.predict(img)
        display_image(f'Input Image',img)
        display_image(f'Predicted Labels for {im}', get_coloured_imgs(op))
        display_image(f'Actual Labels for {im}', get_coloured_imgs(img_seg[:, :, 0]))

if __name__ == "__main__":
    main()
# provide your implementation for the sheet 2 here


