#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#


#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes. 
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats

##*****************************************class 1****************************************
##*****************************************WeakLearner****************************************
#Trains a weak learner from all numerical attribute for all possible split points forpossible feature selection


class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...

    """
    def __init__(self):
        """
        Input:
            

        """
        #print "   "        

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
        '''

        try:
            nexamples, nfeatures = X.shape
        except ValueError:  # Handle cases with a single example or single feature
            nexamples = 1
            nfeatures = X.shape[0] if X.ndim == 1 else 1

        best_score = float('-inf')  # Initialize with the worst possible score
        best_Xlidx = None
        best_Xridx = None
        best_fidx = -1
        best_split = None

        # Loop over each feature to find the best split
        for feat in range(nfeatures):
            # Evaluate the current feature for possible split points
            split, score, Xlidx, Xridx = self.evaluate_numerical_attribute(X[:, feat], Y)
            
            # Update best split if the current split score is better
            if score<best_score:
                best_fidx = feat
                best_split = split
                best_score = score
                best_Xlidx = Xlidx
                best_Xridx = Xridx
    
        self.fidx = best_fidx
        self.split = best_split

        return best_score, best_Xlidx, best_Xridx

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        return X[:, self.fidx] <= self.split


    def evaluate_numerical_attribute(self, feat, Y):
        '''
        Evaluates the numerical attribute for all possible split points for
        possible feature selection. Handles any number of classes.

        Input:
        ---------
        feat: a continuous feature (numpy array)
        Y: labels (numpy array)

        Returns:
        ----------
        v: splitting threshold
        score: splitting score (minimum entropy)
        Xlidx: Index of examples belonging to left child node
        Xridx: Index of examples belonging to right child node
        '''
        best_score = float('inf')  # We want to minimize entropy
        best_split = None
        Xlidx, Xridx = None, None

        # Sort features and corresponding labels
        sidx = np.argsort(feat)
        f = feat[sidx]  # sorted features
        sY = Y[sidx]    # sorted labels

        # Get unique class labels and their counts
        unique_labels, label_counts = np.unique(sY, return_counts=True)
        n_samples = len(sY)

        # Calculate initial entropy of the entire dataset
        class_probabilities = label_counts / n_samples
        H_D = -np.sum(class_probabilities * np.log2(class_probabilities + 1e-9))

        # Find potential split points (midpoints between unique values)
        unique_vals = np.unique(f)
        possible_splits = [(unique_vals[i] + unique_vals[i + 1]) / 2 
                          for i in range(len(unique_vals) - 1)]

        for split_point in possible_splits:
            # Create masks for left and right partitions
            left_mask = f <= split_point
            right_mask = f > split_point

            # Only consider valid splits (both partitions non-empty)
            if np.any(left_mask) and np.any(right_mask):
                left_labels = sY[left_mask]
                right_labels = sY[right_mask]

                # Calculate probabilities and entropy for left partition
                left_counts = np.unique(left_labels, return_counts=True)[1]
                left_probs = left_counts / len(left_labels)
                entropy_left = -np.sum(left_probs * np.log2(left_probs + 1e-9))

                # Calculate probabilities and entropy for right partition
                right_counts = np.unique(right_labels, return_counts=True)[1]
                right_probs = right_counts / len(right_labels)
                entropy_right = -np.sum(right_probs * np.log2(right_probs + 1e-9))

                # Calculate weighted average entropy for this split
                left_weight = len(left_labels) / n_samples
                right_weight = len(right_labels) / n_samples
                split_entropy = (left_weight * entropy_left) + (right_weight * entropy_right)

                # Update best split if current entropy is lower
                if split_entropy < best_score:
                    best_score = split_entropy
                    best_split = split_point
                    Xlidx = sidx[left_mask]  # Convert back to original indices
                    Xridx = sidx[right_mask]

        return best_split, best_score, Xlidx, Xridx


##*****************************************class 2****************************************
##*****************************************RandomWeakLearner****************************************
# Trains a weak learner from random numerical attribute for random split points for possible feature selection

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using 
        a random set of features from the given set of features...
    """
    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...        
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat

    def train(self,X, Y):
        """
        Trains a weak learner from all numerical attribute for all possible split points for
        possible feature selection
        
        Input:
        ---------
        X: a [m x d]  features matrix
        Y: a [m x 1] labels matrix
        
        Returns:
        ----------
        v: splitting threshold
        score: splitting score
        Xlidx: Index of examples belonging to left child node
        Xridx: Index of examples belonging to right child node
        """
        nexamples,nfeatures=X.shape

        best_score = float('-inf')  # Initialize with the worst possible score
        best_Xlidx = None
        best_Xridx = None
        best_fidx = -1
        best_split = None

        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        feature_indices = np.arange(nfeatures)
        selected_features = np.random.choice(feature_indices, size=self.nrandfeat, replace=False)

        # Loop over each feature to find the best split
        for i in selected_features:
            # Evaluate the current feature for possible split points
            split, score, Xlidx, Xridx = self.findBestRandomSplit(X[:, i], Y)
            
            # Update best split if the current split score is better
            if score < best_score:
                best_fidx = i
                best_split = split
                best_score = score
                best_Xlidx = Xlidx
                best_Xridx = Xridx
    
        self.fidx = best_fidx
        self.split = best_split

        return best_score, best_Xlidx,best_Xridx
    
    def findBestRandomSplit(self, feat, Y):
        """
        Find the best random split by randomly sampling "nsplits" splits from the feature range.
        
        Input:
        ----------
        feat: [n X 1] Array of examples for a single feature.
        Y: [n X 1] Array of labels.
        
        Returns:
        ----------
        best_split: The best split threshold found.
        best_score: The minimum entropy achieved by any split.
        Xlidx: Index of examples belonging to the left child node (for the best split).
        Xridx: Index of examples belonging to the right child node (for the best split).
        """
        
        # If nsplits is set to infinity, use evaluate_numerical_attribute for exhaustive search
        if self.nsplits == float('inf'):
            return self.evaluate_numerical_attribute(feat, Y)
        
        # Initialize variables to store the best split found during random sampling
        best_score = float('inf')
        best_split = None
        Xlidx, Xridx = None, None
        
        # Define the range of feature values for random sampling
        min_val, max_val = np.min(feat), np.max(feat)
        
        # Randomly sample nsplits points within the range as potential split points
        random_splits = np.random.uniform(min_val, max_val, self.nsplits)
        
        for split_point in random_splits:
            # Create masks for left and right partitions based on the split point
            left_mask = feat <= split_point
            right_mask = feat > split_point

            # Only consider valid splits (both partitions non-empty)
            if np.any(left_mask) and np.any(right_mask):
                left_labels = Y[left_mask]
                right_labels = Y[right_mask]

                # Calculate probabilities and entropy for left partition
                left_counts = np.unique(left_labels, return_counts=True)[1]
                left_probs = left_counts / len(left_labels)
                entropy_left = -np.sum(left_probs * np.log2(left_probs + 1e-9))

                # Calculate probabilities and entropy for right partition
                right_counts = np.unique(right_labels, return_counts=True)[1]
                right_probs = right_counts / len(right_labels)
                entropy_right = -np.sum(right_probs * np.log2(right_probs + 1e-9))

                # Calculate weighted average entropy for this split
                left_weight = len(left_labels) / len(Y)
                right_weight = len(right_labels) / len(Y)
                split_entropy = (left_weight * entropy_left) + (right_weight * entropy_right)

                # Update best split if current entropy is lower
                if split_entropy < best_score:
                    best_score = split_entropy
                    best_split = split_point
                    Xlidx = np.where(left_mask)[0]
                    Xridx = np.where(right_mask)[0]

        return best_split, best_score, Xlidx, Xridx


    def calculateEntropy(self,Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which 
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl))
        hr= -np.sum(pr*np.log2(pr))

        sentropy = pleft * hl + pright * hr

        return sentropy


##*****************************************class 3****************************************
##*****************************************LinearWeakLearner****************************************
# ax+by+c=0
    
# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
        """
        RandomWeakLearner.__init__(self,nsplits)
        
    def train(self, X, Y):
        '''
        Trains a weak learner by randomly sampling lines (ax + by + c = 0) 
        and finding the best split based on score.
        
        Input:
        ---------
        X: a [m x d] data matrix (numpy array)
        Y: labels (numpy array)
        
        Returns:
        ----------
        best_split: splitting threshold (line parameters a, b, c)
        best_score: splitting score (minimum score achieved)
        best_Xlidx: Index of examples belonging to left child node (for best split)
        best_Xridx: Index of examples belonging to right child node (for best split)
        '''
        nexamples, nfeatures = X.shape
        best_score = float('inf')  # To keep track of the lowest score
        best_split = None
        best_Xlidx, best_Xridx = None, None
        
        # Generate `nsplits` random lines and evaluate them
        for _ in range(self.nsplits):
            # Randomly sample coefficients for the line ax + by + c = 0
            a, b = np.random.randn(2)
            c = np.random.randn()
            
            # Calculate values of ax + by + c for each example
            line_values = a * X[:, 0] + b * X[:, 1] + c
            
            # Split data based on the line's decision boundary
            left_idx = np.where(line_values <= 0)[0]
            right_idx = np.where(line_values > 0)[0]
            
            # Ensure valid split (both sides must have examples)
            if len(left_idx) > 0 and len(right_idx) > 0:
                # Calculate split score (e.g., Gini or entropy)
                left_labels = Y[left_idx]
                right_labels = Y[right_idx]
                
                left_count = np.bincount(left_labels, minlength=len(np.unique(Y)))
                right_count = np.bincount(right_labels, minlength=len(np.unique(Y)))
                
                # Calculate probabilities for each partition
                left_probs = left_count / left_count.sum()
                right_probs = right_count / right_count.sum()
                
                # Calculate entropy for each partition
                entropy_left = -np.sum(left_probs * np.log2(left_probs + 1e-9))
                entropy_right = -np.sum(right_probs * np.log2(right_probs + 1e-9))
                
                # Weighted average of entropy (split score)
                split_score = (len(left_idx) / nexamples) * entropy_left + (len(right_idx) / nexamples) * entropy_right
                
                # Update if the current split is the best
                if split_score < best_score:
                    best_score = split_score
                    best_split = (a, b, c)  # Best line coefficients
                    best_Xlidx = left_idx
                    best_Xridx = right_idx
        self.params = best_split

        return best_score, best_Xlidx, best_Xridx        

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 

        a, b, c = self.params
        line_vals = a * X[:, 0] + b * X[:, 1] + c
        return line_vals <= 0

##*****************************************class 4****************************************
##*****************************************ConicWeakLearner****************************************

class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
    
    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] training matrix...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        

        bfidx=-1, # best feature idx
        minscore=+np.inf
        
        
        tdim= np.ones((nexamples,1))# third dimension
        for i in np.arange(self.nsplits):
            # a*x^2+b*y^2+c*x*y+ d*x+e*y+f


            if i%5==0: # select features indeces after every five iterations
                fidx=np.random.randint(0,nfeatures,2) # sample two random features...
                # Randomly sample a, b and c and test for best parameters...
                parameters = np.random.normal(size=(6,1))
                # apply the line equation...
                res = np.dot ( np.hstack( (np.power(X[:,fidx],2),X[:,fidx],np.prod(X[:,fidx],1)[:,np.newaxis], tdim) ) , parameters )

            splits=np.random.normal(size=(2,1))
            
            # set split to -np.inf for 50% of the cases in the splits...
            if np.random.random(1) < 0.5:
                splits[0]=-np.inf

            tres=  np.logical_and(res >= splits[0], res < splits[1])
            
            score = self.calculateEntropy(Y,tres)

            if score < minscore:
                
                bfidx=fidx # best feature indeces
                bparameters=parameters # best parameters...
                minscore=score
                bres= tres
                bsplits=splits

        self.parameters=bparameters
        self.score=minscore
        self.splits=bsplits
        self.fidx=bfidx
        
        bXl=np.squeeze(bres)
        bXr=np.logical_not(bXl)

        return minscore, bXl, bXr

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        fidx=self.fidx
        res = np.dot ( np.hstack( (np.power(X[:,fidx],2),X[:,fidx],np.prod(X[:,fidx],1)[:,np.newaxis], np.ones((X.shape[0],1))) ) , self.parameters ) 
        return np.logical_and(res >= self.splits[0], res < self.splits[1])
"""    
wl=WeakLearner()
rwl=RandomWeakLearner()
lwl=LinearWeakLearner()
"""
        
