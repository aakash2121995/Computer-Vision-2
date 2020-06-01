import cv2 as cv
import numpy as np
import random

from custom_hog_detector import CustomHogDetector
# Global constants

# crop/patch dimensions for the training samples
width = 64
height = 128

num_negative_samples = 10 # number of negative samples per image
train_hog_path = '../train_hog_descs.npy' # the file to which you save the HOG descriptors of every patch
train_labels = '../labels_train.npy' # the file to which you save the labels of the training data
my_svm_filename = '../my_pretrained_svm.dat' # the file to which you save the trained svm

#data paths
test_images_1 = '../data/task_1_testImages/'
path_train_2 = '../data/task_2_3_Data/train/'
path_test_2 = '../data/task_2_3_Data/test/'

#***********************************************************************************
# draw a bounding box in a given image
# Parameters:
# im: The image on which you want to draw the bounding boxes
# detections: the bounding box of the detections (people)
# returns None

def drawBoundingBox(im, detections):
    for x, y, w, h in detections:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv.rectangle(im, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)
    cv.imshow('Detections', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

def task1():
    print('Task 1 - OpenCV HOG')

    # Load images

    filelist = test_images_1 + 'filenames.txt'
    test_img = []
    with open(filelist) as f:
        for path in f.readlines():
            test_img.append(cv.imread(test_images_1 + path.strip()))

    # TODO: Create a HOG descriptor object to extract the features and detect people. Do this for every
    #       image, then draw a bounding box and display the image
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    for img in test_img:
        found, _w = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32), scale=1.05)
        drawBoundingBox(img,found)
    cv.waitKey(0)
    cv.destroyAllWindows()


def task2():

    print('Task 2 - Extract HOG features')

    random.seed()
    np.random.seed()

    # Load image names

    filelist_train_pos = path_train_2 + 'filenamesTrainPos.txt'
    filelist_train_neg = path_train_2 + 'filenamesTrainNeg.txt'

    train_data = []
    labels = []
    hog = cv.HOGDescriptor()

    with open(filelist_train_pos) as f:
        for path in f.readlines():
            img = cv.imread(path_train_2 + 'pos/' + path.strip())
            center = img.shape[0]//2,img.shape[1]//2
            img = img[center[0] - height//2:center[0] + height//2,center[1] - width//2:center[1] + width//2 ]
            descriptor = hog.compute(img)
            train_data.append(descriptor.squeeze())
            labels.append(1)

    with open(filelist_train_neg) as f:
        for path in f.readlines():
            img = cv.imread(path_train_2 + 'neg/' + path.strip())
            for i in range(10):
                # print(img.shape)
                x,y = np.random.randint(0,img.shape[1]-width),np.random.randint(0,img.shape[0]-height)
                patch = img[y:y + height, x:x + width]
                descriptor = hog.compute(patch)
                train_data.append(descriptor.squeeze())
                labels.append(0)
    print('Saving Data ...')
    np.save(train_labels,labels)
    np.save(train_hog_path,train_data)


    # TODO: Create a HOG descriptor object to extract the features from the set of positive and negative samples

    # positive samples: Get a crop of size 64*128 at the center of the image then extract its HOG features
    # negative samples: Sample 10 crops from each negative sample at random and then extract their HOG features
    # In total you should have  (x+10*y) training samples represented as HOG features(x=number of positive images, y=number of negative images),
    # save them and their labels in the path train_hog_path and train_labels in order to load them in section 3







def task3():
    print('Task 3 - Train SVM and predict confidence values')
      #TODO Create 3 SVMs with different C values, train them with the training data and save them
      # then use them to classify the test images and save the results
    print('Loading Training Data ..')
    trainingData = np.load(train_hog_path)
    tr_labels = np.load(train_labels)

    filelist_testPos = path_test_2 + 'filenamesTestPos.txt'
    filelist_testNeg = path_test_2 + 'filenamesTestNeg.txt'

    hog = cv.HOGDescriptor()
    test_data = []
    tst_labels = []

    print('Loading Test Data ..')

    with open(filelist_testPos) as f:
        for path in f.readlines():
            img = cv.imread(path_test_2 + 'pos/' + path.strip())
            center = img.shape[0]//2,img.shape[1]//2
            img = img[center[0] - height//2:center[0] + height//2,center[1] - width//2:center[1] + width//2 ]
            descriptor = hog.compute(img)
            test_data.append(descriptor.squeeze())
            tst_labels.append(1)

    with open(filelist_testNeg) as f:
        for path in f.readlines():
            img = cv.imread(path_test_2 + 'neg/' + path.strip())
            for y in range(0,img.shape[0]-height,height):
                for x in range(0,img.shape[1]-width,width):
                    patch = img[y:y+height,x:x+width]
                    descriptor = hog.compute(patch)
                    test_data.append(descriptor.squeeze())
                    tst_labels.append(0)
    test_data = np.stack(test_data)
    tst_labels = np.array(tst_labels)

    for C in [0.01,1,100]:
        print(f'Training Model C = {C}')
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setKernel(cv.ml.SVM_LINEAR)
        svm.setC(C)
        svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        svm.train(trainingData, cv.ml.ROW_SAMPLE, tr_labels)
        confidence_score = svm.predict(test_data,svm.predict(test_data)[1],cv.ml.StatModel_RAW_OUTPUT)

        print(f'Saving Model and confidence scores and test labels for C = {C}')
        np.save(f'confidence_{C}.npy',confidence_score[1])
        svm.save(f'SVM_{C}_.dat')

    print('Saving Test Labels ..')
    np.save('test_labels.npy',tst_labels)






def task5():

    print ('Task 5 - Eliminating redundant Detections')


    # TODO: Write your own custom class myHogDetector
    # Note: compared with the previous tasks, this task requires more coding

    my_detector = CustomHogDetector(my_svm_filename)

    # TODO Apply your HOG detector on the same test images as used in task 1 and display the results

    print('Done!')
    cv.waitKey()
    cv.destroyAllWindows()






if __name__ == "__main__":

    # Task 1 - OpenCV HOG
    task1()

    # Task 2 - Extract HOG Features
    task2()

    # Task 3 - Train SVM
    task3()
    #
    # # Task 5 - Multiple Detections
    # task5()

