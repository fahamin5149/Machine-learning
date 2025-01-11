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

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:
            

        """
        #print "   "        
        pass

    def train(self,X, Y):
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
        #nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        try:
            nexamples, nfeatures=X.shape
        except:
            nexamples=1
        #---------End of Your Code-------------------------#
        #return score, Xlidx,Xridx
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#            
        
        #---------End of Your Code-------------------------#
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
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
        
        #classes=np.unique(Y)
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...
        
        #---------End of Your Code-------------------------#
            
        #return split,mingain,Xlidx,Xridx

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
        pass

    def train(self,X, Y):
        '''
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
            
        '''
        nexamples,nfeatures=X.shape

        
        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        #---------End of Your Code-------------------------#
        #return minscore, bXl,bXr

    def findBestRandomSplit(self,feat,Y):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        return splitvalue, score, np.array(indxLeft), indxRight

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

            
        
        #---------End of Your Code-------------------------#
        #return splitvalue, minscore, Xlidx, Xridx
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
        
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        
        #---------End of Your Code-------------------------#
        return scores[minindx],xlind,xrind
        #return minscore, bXl, bXr
        

    

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        

        #---------End of Your Code-------------------------#
        # build a classifier ax+by+c=0
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
        
        pass

    
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
        
