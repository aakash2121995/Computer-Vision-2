import numpy as np
from Node import  Node


class DecisionTree():
    def __init__(self, patches, labels, tree_param):

        self.patches, self.labels = patches, labels
        self.depth = tree_param['depth']
        self.pixel_locations = tree_param['pixel_locations']
        self.random_color_values = tree_param['random_color_values']
        self.no_of_thresholds = tree_param['no_of_thresholds']
        self.minimum_patches_at_leaf = tree_param['minimum_patches_at_leaf']
        self.classes = tree_param['classes']
        self.nodes = []

    # Function to train the tree
    # provide your implementation
    # should return a trained tree with provided tree param
    def train(self):
        root = self.build_tree(self.patches,self.labels)
        self.nodes.append(root)

    # Function to predict probabilities for single image
    # provide your implementation
    # should return predicted class for every pixel in the test image
    def predict(self, I):
        patch_size = self.patches.shape[1]
        output = np.zeros((I.shape[0]-patch_size+1,I.shape[1]-patch_size+1))
        for i in range(I.shape[0]-patch_size):
            for j in range(I.shape[1] - patch_size):
                node = self.nodes[-1] # root
                patch = I[i:i+patch_size, j:j+patch_size]
                while node.type == 'SPLIT':
                    feature = node.feature
                    i,j = feature['pixel_location'][0], feature['pixel_location'][1]
                    channel = feature['color']
                    if patch[i,j,channel] < feature['th']:
                        node = node.leftChild
                    else:
                        node = node.rightChild
                output[i,j] = node.probabilities.argmax()

        return output

    # Function to get feature response for a random color and pixel location
    # provide your implementation
    # should return feature response for all input patches
    def getFeatureResponse(self, patches, feature):
        i,j = feature['pixel_location'][0],feature['pixel_location'][1]
        values = patches[:,i,j,feature['color']]
        return values

    # Function to get left/right split given feature responses and a threshold
    # provide your implementation
    # should return left/right split
    def getsplit(self, responses, threshold):
        left_flag = responses < threshold
        left = np.arange(responses.shape[0])[left_flag]
        right = np.arange(responses.shape[0])[~left_flag]
        return left,right
    # Function to get a random pixel location
    # provide your implementation
    # should return a random location inside the patch
    def generate_random_pixel_location(self):
        return np.random.randint(0,16),np.random.randint(0,16)

    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, labels):
        class_, count = np.unique(labels, return_counts=True)
        probs = count/count.sum()
        log_prob = np.log(count)
        return -np.dot(log_prob,probs)

    # Function to measure information gain for a given split
    # provide your implementation
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        return EntropyAll - (Entropyleft*Nleft + Entropyright*Nright)/Nall

    # Function to get the best split for given patches with labels
    # provide your implementation
    # should return left,right split, color, pixel location and threshold
    def best_split(self, patches, labels):
        best_gain = 0
        best_feature = None
        entropy_all = self.compute_entropy(labels)
        Nall = patches.shape[0]
        for c in range(self.random_color_values):
            channel = np.random.randint(3)
            for p in range(self.pixel_locations):
                pixel_location = self.generate_random_pixel_location()
                for t in range(self.no_of_thresholds):
                    threshhold = 255 * np.random.rand()

                feature = self.get_feature(channel,pixel_location,threshhold)
                responses = self.getFeatureResponse(patches,feature)
                left_ids,right_ids = self.getsplit(responses,feature['th'])
                left = labels[left_ids]
                right = labels[right_ids]
                entropy_left = self.compute_entropy(left)
                entropy_right = self.compute_entropy(right)
                Nleft, Nright = left.shape[0],left.shape[1]
                gain = self.get_information_gain(entropy_left,entropy_right,entropy_all, Nall,Nleft, Nright)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature

        responses = self.getFeatureResponse(patches, best_feature)
        left,right = self.getsplit(responses, feature['th'])
        return left,right,best_feature


    # feel free to add any helper functions
    def get_feature(self,pixel_loc,channel,threshhold):
        return  {'color': channel, 'pixel_location': pixel_loc, 'th': threshhold}

    def build_tree(self,patches,labels,depth=0):
        if depth >= self.depth:
            return None
        left_id, right_id,feature = self.best_split(patches,labels)
        left_patches, left_labels = patches[left_id], labels[left_id]
        right_patches, right_labels = patches[right_id], labels[right_id]
        node = Node()

        if depth == self.depth - 1 or left_patches.shape[0] < self.minimum_patches_at_leaf or right_patches.shape[0] < self.minimum_patches_at_leaf:
            node.create_leafNode(patches,self.classes)
            self.nodes.append(node)
            return node
        else:
            left = self.build_tree(left_patches,left_labels,depth+1)
            right = self.build_tree(right_patches,right_labels,depth+1)
            node.create_SplitNode(left,right,feature)
            self.nodes.append(node)
            return node








