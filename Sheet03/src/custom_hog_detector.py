
    # TODO: This class should which Implements the following functionality:
    # - opencv HOGDescriptor combined with a sliding window
    # - perform detection at multiple scales, i.e. you need to scale the extracted patches when performing the detection
    # - non maximum suppression: eliminate detections using non-maximum-suppression based on the overlap area
import numpy as np
import time
import cv2 as cv
# from src.sheet4 import drawBoundingBox
class CustomHogDetector:

    # Some constants that you will be using in your implementation
    detection_width	= 64 # the crop width dimension
    detection_height = 128 # the crop height dimension
    window_stride = 32 # the stride size 
    scaleFactors = [1.2] # scale each patch down by this factor, feel free to try other values
    # You may play with different values for these two theshold values below as well
    hit_threshold = 0 # detections above this threshold are counted as positive. 
    overlap_threshold = 0.3 # if the overlap between two detections is above this threshold, eliminate the one with the lower confidence score. 

    def __init__(self, trained_svm_name):
        #load the trained SVM from file trained_svm_name
        self.svm = cv.ml.SVM_load(trained_svm_name)

    def drawBoundingBoxAndSave(self, im, detections, img, type):
        for x, y, w, h, s in detections:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15 * w), int(0.01 * h)
            cv.rectangle(im, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)
        image_title = img + type+".png"
        #to save the images
        # cv.imwrite(image_title, im)
        cv.imshow('Detections'+type, im)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def non_max_suppression_fast(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        boxes = b = (np.array(boxes, dtype=np.float))
        pick = []
        x1 = y1 = x2 = y2 = np.array([])
        # grab the coordinates of the bounding boxes
        scores = boxes[:, 4]
        i=0
        for score in (scores):
            if abs(score) > 0.0:
                x1 = np.append(x1,boxes[i][0])
                y1 = np.append(y1,boxes[i][1])
                x2 = np.append(x2,boxes[i][2] + x1[i])
                y2 = np.append(y2,boxes[i][3] + y1[i])
                i +=1
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    def pyramid(self, image, scale=1.2, minSize=(64, 128)):
        # yield the original image
        yield image
        # keep looping over the pyramid
        while True:
            # compute the new dimensions of the image and resize it
            w = int(image.shape[1] / scale)
            h = int(image.shape[0] / scale)
            image = cv.resize(image,(w,h))
            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break
            # yield the next image in the pyramid
            yield image

    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


    def detector(self, test_img, imgs, show_sliding_window = False):
        for image,img in zip(test_img,imgs):
            (winW, winH) = (self.detection_width,self.detection_height )
            prediction_window = []
            factor = 0
            for resized in self.pyramid(image, scale = 1.5):

                # loop over the sliding window for each layer of the pyramid
                for (x, y, window) in self.sliding_window(resized, stepSize = 32, windowSize = (winW, winH)):
                    # if the window does not meet our desired window size, ignore it
                    if window.shape[0] != winH or window.shape[1] != winW:
                        continue

                    hog = cv.HOGDescriptor()
                    descriptor = (hog.compute(window)).squeeze()
                    descriptor = np.reshape(descriptor,(1,descriptor.shape[0]))
                    found =  self.svm.predict(descriptor)[1]

                    if found[0][0] == 1:
                        confidence_score = self.svm.predict(descriptor, self.svm.predict(descriptor)[1], cv.ml.StatModel_RAW_OUTPUT)[1].squeeze()
                        int((np.power(1.5, factor)))
                        mult = np.power(1.5, factor)
                        prediction_window.append([int(x*mult), int(y*mult), int(self.detection_width*mult), int( self.detection_height*mult), confidence_score.item()])
                    if show_sliding_window:
                        clone = resized.copy()
                        cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                        cv.imshow("sliding window and detection at multiple scales", clone)
                        cv.waitKey(1)
                        time.sleep(0.15)
                factor = factor + 1
            self.drawBoundingBoxAndSave(image.copy(), prediction_window,img, type="Without_NMS")
            pick = self.non_max_suppression_fast(prediction_window, overlapThresh=self.overlap_threshold)
            self.drawBoundingBoxAndSave(image.copy(), pick, img, "With_NMS")