import numpy as np


class Node():
    def __init__(self):

        self.type = 'None'
        self.leftChild = -1
        self.rightChild = -1
        self.feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}
        self.probabilities = []


    # Function to create a new split node
    # provide your implementation
    def create_SplitNode(self, leftchild, rightchild, feature):
        self.type = 'SPLIT'
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.feature = feature

    # Function to create a new leaf node
    # provide your implementation
    def create_leafNode(self, labels, classes):
        self.probabilities = np.zeros((len(classes)))
        self.type = 'LEAF'
        class_, count = np.unique(labels, return_counts=True)

        cls_to_index = {cls:i for i,cls in enumerate(classes)}

        for cls,count in zip(class_, count):
            self.probabilities[cls_to_index[cls]] = count

        self.probabilities /= count.sum()



    # feel free to add any helper functions