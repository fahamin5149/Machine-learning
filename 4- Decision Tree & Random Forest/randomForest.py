#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

#---------------Instructions------------------#
# Please read the function documentation before
# proceeding with code writing. 

# For randomizing, you will need to use following functions
# please refer to their documentation for further help.
# 1. np.random.randint
# 2. np.random.random
# 3. np.random.shuffle
# 4. np.random.normal 


# Other Helpful functions: np.atleast_2d, np.squeeze()
# scipy.stats.mode, np.newaxis

#-----------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#
import tree as tree_attrib
import numpy as np
import scipy.stats as stats
from scipy.stats import mode


class RandomForest:
    ''' Implements the Random Forest For Classification... '''
    def __init__(self, ntrees=10,treedepth=5,usebagging=False,baggingfraction=0.6,
        weaklearner="Conic",
        nsplits=10,        
        nfeattest=None, posteriorprob=False,scalefeat=True ):        
        """      
            Build a random forest classification forest....

            Input:
            ---------------
                ntrees: number of trees in random forest
                treedepth: depth of each tree 
                usebagging: to use bagging for training multiple trees
                baggingfraction: what fraction of training set to use for building each tree,
                weaklearner: which weaklearner to use at each interal node, e.g. "Conic, Linear, Axis-Aligned, Axis-Aligned-Random",
                nsplits: number of splits to test during each feature selection round for finding best IG,                
                nfeattest: number of features to test for random Axis-Aligned weaklearner
                posteriorprob: return the posteriorprob class prob 
                scalefeat: wheter to scale features or not...
        """

        self.ntrees=ntrees
        self.treedepth=treedepth
        self.usebagging=usebagging
        self.baggingfraction=baggingfraction

        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.nfeattest=nfeattest
        
        self.posteriorprob=posteriorprob
        
        self.scalefeat=scalefeat
        
        self.trees = []

    def findScalingParameters(self,X):
        """
            find the scaling parameters
            input:
            -----------------
                X= m x d training data matrix...
        """
        means = np.mean(X, axis=0)
        std_devs = np.std(X, axis=0)

        # Store or return scaling parameters
        self.scaling_params = {'means': means, 'std_devs': std_devs}
        
        return means, std_devs
    
    def applyScaling(self,X):
        """
            Apply the scaling on the given training parameters
            Input:
            -----------------
                X: m x d training data matrix...
            Returns:
            -----------------
                X: scaled version of X
        """
        # Ensure that scaling parameters have been calculated
        if not hasattr(self, 'scaling_params'):
            raise ValueError("Scaling parameters not found. Call `findScalingParameters` first.")
        
        # Retrieve stored mean and standard deviation for each feature
        means = self.scaling_params['means']
        std_devs = self.scaling_params['std_devs']
        
        # Apply scaling: (X - mean) / std_dev
        X_scaled = (X - means) / std_devs
        
        return X_scaled

    def train(self,X,Y,vX=None,vY=None):
        '''
        Trains a RandomForest using the provided training set..
        
        Input:
        ---------
        X: a m x d matrix of training data...
        Y: labels (m x 1) label matrix

        vX: a n x d matrix of validation data (will be used to stop growing the RF)...
        vY: labels (n x 1) label matrix

        Returns:
        -----------

        '''
        import tools as t

        nexamples, nfeatures= X.shape

        self.findScalingParameters(X)
        if self.scalefeat:
            X=self.applyScaling(X)

        self.trees=[]

        # self, ntrees=10, treedepth=5, usebagging=False, baggingfraction=0.6, weaklearner="Conic",
        # nsplits=10,nfeattest=None, posteriorprob=False, scalefeat=True

        for tree in range(self.ntrees):
            if self.usebagging:
                sample_size = int(self.baggingfraction * nexamples)
                indices = np.random.choice(nexamples, sample_size, replace=True)
                sample_X = X[indices]
                sample_Y = Y[indices]
            else:
                sample_X = X
                sample_Y = Y

                        # Create and train new tree
            newtree = tree_attrib.DecisionTree(
                exthreshold=5,
                maxdepth=self.treedepth,
                weaklearner=self.weaklearner,
                pdist=self.posteriorprob,
                nsplits=self.nsplits,
                nfeattest=self.nfeattest
            )
            newtree.train(sample_X, sample_Y)
            self.trees.append(newtree)

import numpy as np

# Function to compute the transformed features (polynomial degree 2)
def transform_features(X):
    X_transformed = np.ones((X.shape[0], 6))  # [1, x, y, x^2, y^2, xy]
    X_transformed[:, 1] = X[:, 0]  # x
    X_transformed[:, 2] = X[:, 1]  # y
    X_transformed[:, 3] = X[:, 0] ** 2  # x^2
    X_transformed[:, 4] = X[:, 1] ** 2  # y^2
    X_transformed[:, 5] = X[:, 0] * X[:, 1]  # xy
    return X_transformed

# Function to compute the hypothesis
def hypothesis(X, theta):
    return np.dot(X, theta)

# Function to compute the cost function (Mean Squared Error)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent function to optimize theta
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = hypothesis(X, theta)
        gradients = (1 / m) * np.dot(X.T, (predictions - y))  # Compute gradient
        theta = theta - alpha * gradients  # Update theta
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

    def predict(self, X):
        
        """
        Test the trained RF on the given set of examples X
        
                   
            Input:
            ------
                X: [m x d] a d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        z=[]
        
        if self.scalefeat:
            X=self.applyScaling(X)
        
        predictions = np.array([tree.predict(X) for tree in self.trees])

        if len(predictions.shape) > 2:
            predictions = np.squeeze(predictions)
        final_predictions = mode(predictions, axis=0).mode[0]

        return final_predictions